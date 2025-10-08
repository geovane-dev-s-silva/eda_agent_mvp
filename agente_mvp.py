from flask import Flask, request, jsonify, send_file
import os, io, uuid, time, json, base64
import pandas as pd
from eda_agent import (
    load_csv_bytes, quick_summary, save_dataset_metadata, save_query, infer_schema, DB_CONN,
    histogram_plot, boxplot_plot, correlation_heatmap, detect_outliers_iqr
)
from call_gemini import call_gemini
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# new helpers
from agent_autoinsight import generate_insights
from agent_memory import save_memory, load_memory

app = Flask(__name__)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "files" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    files = request.files.getlist("files")
    dfs = []
    filenames = []
    for f in files:
        content = f.read()
        df = load_csv_bytes(content)
        df.columns = df.columns.str.strip()
        dfs.append(df)
        filenames.append(f.filename)
    if len(dfs) == 1:
        df_combined = dfs[0]
    else:
        # try merge or concat
        common = set(dfs[0].columns)
        for d in dfs[1:]:
            common = common.intersection(set(d.columns))
        if common:
            key = list(common)[0]
            df_combined = dfs[0]
            for d in dfs[1:]:
                df_combined = df_combined.merge(d, on=key, how="outer")
        else:
            df_combined = pd.concat(dfs, ignore_index=True, sort=False)

    dataset_id = str(uuid.uuid4())
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    df_combined.to_csv(path, index=False)
    save_dataset_metadata(dataset_id, ",".join(filenames), df_combined, path)
    summary = quick_summary(df_combined)

    # Generate auto-insights asynchronously would be ideal; for now run synchronously
    ai = generate_insights(dataset_id, df_combined)

    response = {
        "dataset_id": dataset_id,
        "n_rows": df_combined.shape[0],
        "n_cols": df_combined.shape[1],
        "schema": summary["schema"],
        "plots": summary["plots"],
        "auto_insights_preview": ai["insights"][:3]
    }
    return jsonify(response), 200

@app.route("/api/insights", methods=["GET"])
def get_insights():
    dataset_id = request.args.get("dataset_id")
    cur = DB_CONN.cursor()
    rows = cur.execute("SELECT insight_id, text, important, created_at FROM insights WHERE dataset_id=? ORDER BY created_at DESC", (dataset_id,)).fetchall()
    insights = [{"id": r[0], "text": r[1], "important": bool(r[2]), "created_at": r[3]} for r in rows]
    return jsonify(insights), 200

@app.route("/api/insights/mark", methods=["POST"])
def mark_insight():
    data = request.get_json()
    iid = data.get("insight_id")
    cur = DB_CONN.cursor()
    cur.execute("UPDATE insights SET important=1 WHERE insight_id=?", (iid,))
    DB_CONN.commit()
    return jsonify({"status": "ok"}), 200

# query with memory context
@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "dataset_id" not in data or "question" not in data:
        return jsonify({"error": "Campos dataset_id e question são obrigatórios"}), 400
    dataset_id = data["dataset_id"]
    question = data["question"]
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset_id não encontrado"}), 404
    df = pd.read_csv(path)

    # try pandas quick matches
    import re, numpy as np
    patterns = {
        "mean": r"(m[eé]dia|m[eé]dio) da coluna (\w+)",
        "max": r"(maior|m[aá]ximo) da coluna (\w+)",
        "min": r"(menor|m[ií]nimo) da coluna (\w+)",
        "sum": r"(soma|total) da coluna (\w+)",
        "count": r"(quantidade|contagem|n[úu]mero de) (\w+)",
    }
    for op, pat in patterns.items():
        m = re.search(pat, question, flags=re.IGNORECASE)
        if m:
            col = m.group(len(m.groups()))
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                val = None
                if op == "mean": val = df[col].mean()
                elif op == "max": val = df[col].max()
                elif op == "min": val = df[col].min()
                elif op == "sum": val = df[col].sum()
                elif op == "count": val = df[col].count()
                answer = f"A {op} da coluna {col} é {val:.2f}."
                save_query(dataset_id, question, answer, answer, "pandas")
                # save to memory
                save_memory(dataset_id, question, answer)
                return jsonify({"answer": answer, "source": "pandas"}), 200

    # fallback to LLM with memory context
    schema = infer_schema(df)
    stats = df.describe(include="all").to_dict()
    history = load_memory(dataset_id, limit=5)
    history_text = ""
    if history:
        history_text = "\n".join([f"Usuário: {q}\nAgente: {a}" for q,a in history])
    context = {"schema": schema, "stats": stats, "history": history_text}
    prompt = f"""Você é um analista de dados experiente. Use apenas o contexto fornecido.
Contexto: {json.dumps(context)[:6000]}
Pergunta: {question}
Responda em português, sendo objetivo e citando números quando possível."""
    try:
        llm_answer = call_gemini(prompt)
        save_query(dataset_id, question, llm_answer, llm_answer, "llm")
        save_memory(dataset_id, question, llm_answer)
        return jsonify({"answer": llm_answer, "source": "llm"}), 200
    except Exception as e:
        return jsonify({"error": f"Falha ao chamar LLM: {str(e)}"}), 500

if __name__ == "__main__":
    print("Rodando Flask app: python agente_mvp.py")
    app.run(host="0.0.0.0", port=8000)
