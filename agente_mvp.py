# --- Importações principais ---
from flask import Flask, request, jsonify, send_file  # Framework web
import os, io, uuid, time, json, base64  # Utilidades
import pandas as pd  # Manipulação de dados
from eda_agent import (
    load_csv_bytes,
    quick_summary,
    save_dataset_metadata,
    save_query,
    detect_outliers_iqr,
    correlation_heatmap,
    boxplot_plot,
)  # Funções utilitárias do projeto
from call_gemini import call_gemini  # Integração com LLM Gemini
from sklearn.cluster import KMeans  # Clustering
import matplotlib.pyplot as plt  # Gráficos

# --- Inicialização do app Flask e diretórios ---
app = Flask(__name__)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)  # Garante que a pasta de dados existe


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """
    Endpoint para upload de um ou mais arquivos CSV.
    Salva, concatena, infere esquema e gera resumo inicial.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    # Lê e concatena todos os arquivos enviados
    all_dfs = []
    filenames = []
    for f in files:
        content = f.read()
        df = load_csv_bytes(content)
        all_dfs.append(df)
        filenames.append(f.filename)
    df_combined = pd.concat(all_dfs, ignore_index=True)
    dataset_id = str(uuid.uuid4())
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    df_combined.to_csv(path, index=False)
    # Salva metadados do dataset
    save_dataset_metadata(dataset_id, ",".join(filenames), df_combined, path)

    # Gera resumo automático
    summary = quick_summary(df_combined)
    response = {
        "dataset_id": dataset_id,
        "n_rows": df_combined.shape[0],
        "n_cols": df_combined.shape[1],
        "schema": summary["schema"],
        "plots": summary["plots"],
    }
    return jsonify(response)


@app.route("/api/query", methods=["POST"])
def query():
    """
    Endpoint para perguntas em linguagem natural sobre o dataset.
    Tenta responder com Pandas (regex simples) ou LLM (Gemini).
    Salva histórico de perguntas e respostas.
    """
    data = request.get_json()
    if not data or "dataset_id" not in data or "question" not in data:
        return jsonify({"error": "Campos obrigatórios ausentes"}), 400

    dataset_id = data["dataset_id"]
    question = data["question"]
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset_id não encontrado"}), 404
    df = pd.read_csv(path)

    import re, numpy as np

    # Regex para identificar perguntas simples (média, soma, etc.)
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
            # Se a coluna existe e é numérica, responde direto com Pandas
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                val = None
                if op == "mean":
                    val = df[col].mean()
                elif op == "max":
                    val = df[col].max()
                elif op == "min":
                    val = df[col].min()
                elif op == "sum":
                    val = df[col].sum()
                elif op == "count":
                    val = df[col].count()
                answer = f"A {op} da coluna {col} é {val:.2f}."
                save_query(dataset_id, question, answer, answer, "pandas")
                return jsonify({"answer": answer, "source": "pandas"})

    # Caso não seja uma pergunta simples, chama o LLM (Gemini)
    from eda_agent import infer_schema

    schema = infer_schema(df)
    context = {"schema": schema, "sample": df.head(5).to_dict()}
    prompt = f"Pergunta: {question}\nContexto: {json.dumps(context)[:4000]}"
    try:
        llm_answer = call_gemini(prompt)
        save_query(dataset_id, question, llm_answer, llm_answer, "llm")
        return jsonify({"answer": llm_answer, "source": "llm"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outliers", methods=["GET"])
def get_outliers():
    """
    Endpoint para detecção de outliers em uma coluna numérica.
    Retorna estatísticas e boxplot em base64.
    """
    dataset_id = request.args.get("dataset_id")
    col = request.args.get("col")
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset não encontrado"}), 404
    df = pd.read_csv(path)
    if col not in df.columns:
        return jsonify({"error": "coluna não encontrada"}), 400
    stats = detect_outliers_iqr(df[col].dropna())
    plot_b64 = boxplot_plot(df, col)
    return jsonify({"stats": stats, "plot": plot_b64})


@app.route("/api/correlation", methods=["GET"])
def get_correlation():
    """
    Endpoint para análise de correlação entre colunas numéricas.
    Retorna matriz, heatmap e insight automático.
    """
    dataset_id = request.args.get("dataset_id")
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset não encontrado"}), 404
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return jsonify({"error": "colunas insuficientes"}), 400
    plot_b64, corr = correlation_heatmap(df, num_cols)
    return jsonify({"corr": corr, "plot": plot_b64})


@app.route("/api/report", methods=["GET"])
def report():
    """
    Endpoint para exportação de relatório PDF consolidado.
    Inclui resumo, esquema, insights e gráficos principais.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    dataset_id = request.args.get("dataset_id")
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset não encontrado"}), 404
    df = pd.read_csv(path)
    summary = quick_summary(df)
    pdf_path = os.path.join(DATA_DIR, f"{dataset_id}_report.pdf")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)
    flow = [Paragraph(f"Relatório Dataset {dataset_id}", styles["Title"])]
    flow.append(Spacer(1, 12))
    flow.append(
        Paragraph(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}", styles["Normal"])
    )
    doc.build(flow)
    return send_file(pdf_path, as_attachment=True)


# --- Inicialização do servidor Flask ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
