from flask import Flask, request, jsonify, send_file
import os, io, uuid, time
import pandas as pd
import json
import re
import numpy as np
import base64
import matplotlib.pyplot as plt
from eda_agent import save_query, infer_schema
from call_gemini import call_gemini
from eda_agent import load_csv_bytes, quick_summary, save_dataset_metadata

from sklearn.cluster import KMeans
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
DATA_DIR = os.environ.get("EDA_DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# Função utilitária para outliers (IQR)
def detect_outliers_iqr(series):
    """
    Detecta outliers em uma série numérica usando o método do IQR (Intervalo Interquartil).
    Retorna um dicionário com contagem, limites e exemplos de outliers.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "count": int(outliers.count()),
        "lower": float(lower),
        "upper": float(upper),
        "examples": outliers.head(5).tolist(),
    }


def boxplot_plot(df, col):
    """
    Gera um boxplot em base64 para a coluna informada do DataFrame.
    """
    fig, ax = plt.subplots()
    ax.boxplot(df[col].dropna(), vert=True)
    ax.set_title(f"Boxplot de {col}")
    ax.set_ylabel(col)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


@app.route("/api/outliers", methods=["GET"])
def get_outliers():
    """
    Endpoint GET /api/outliers
    Retorna estatísticas de outliers e boxplot para uma coluna numérica do dataset.
    """
    dataset_id = request.args.get("dataset_id")
    col = request.args.get("col")
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset não encontrado"}), 404
    df = pd.read_csv(path)
    if col not in df.columns:
        return jsonify({"error": "coluna não encontrada"}), 400
    if not pd.api.types.is_numeric_dtype(df[col]):
        return jsonify({"error": "coluna não é numérica"}), 400
    stats = detect_outliers_iqr(df[col].dropna())
    box_b64 = boxplot_plot(df, col)
    return jsonify({"stats": stats, "plot": box_b64})


def correlation_heatmap(df, cols):
    """
    Gera um heatmap de correlação entre as colunas numéricas informadas.
    Retorna a imagem em base64 e o dicionário de correlação.
    """
    corr = df[cols].corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_yticklabels(cols)
    fig.colorbar(im)
    ax.set_title("Heatmap de Correlação")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}", corr.to_dict()


@app.route("/api/correlation", methods=["GET"])
def get_correlation():
    """
    Endpoint GET /api/correlation
    Retorna matriz de correlação, heatmap e insight automático das colunas mais correlacionadas.
    """
    dataset_id = request.args.get("dataset_id")
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset não encontrado"}), 404
    df = pd.read_csv(path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return jsonify({"error": "dados insuficientes"}), 400
    plot_b64, corr_dict = correlation_heatmap(df, numeric_cols)
    # Insight automático
    corr_matrix = df[numeric_cols].corr().abs()
    arr = corr_matrix.values
    arr[np.tril_indices_from(arr)] = 0
    max_idx = np.unravel_index(np.argmax(arr), arr.shape)
    max_val = arr[max_idx]
    insight = f"Colunas {numeric_cols[max_idx[0]]} e {numeric_cols[max_idx[1]]} têm a maior correlação (r={max_val:.2f})"
    return jsonify({"corr": corr_dict, "plot": plot_b64, "insight": insight})


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    """
    Endpoint GET /api/clusters
    Realiza clustering KMeans em duas colunas numéricas e retorna o gráfico e os clusters.
    """
    dataset_id = request.args.get("dataset_id")
    cols = request.args.get("cols")
    k = int(request.args.get("k", 3))
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset não encontrado"}), 404
    df = pd.read_csv(path)
    if not cols:
        return jsonify({"error": "parâmetro 'cols' é obrigatório"}), 400
    col_list = cols.split(",")
    if len(col_list) != 2 or not all(c in df.columns for c in col_list):
        return jsonify({"error": "colunas inválidas"}), 400
    sub = df[col_list].dropna()
    if sub.shape[0] < k:
        return jsonify({"error": "dados insuficientes para clustering"}), 400
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(sub)
    sub["cluster"] = km.labels_
    fig, ax = plt.subplots()
    colors = plt.cm.get_cmap("tab10", k)
    for label in range(k):
        cluster_data = sub[sub["cluster"] == label]
        ax.scatter(
            cluster_data[col_list[0]],
            cluster_data[col_list[1]],
            label=f"Cluster {label}",
            color=colors(label),
        )
    ax.set_xlabel(col_list[0])
    ax.set_ylabel(col_list[1])
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    insight = f"Foram identificados {k} clusters distintos no espaço ({col_list[0]},{col_list[1]})."
    return jsonify(
        {
            "clusters": sub["cluster"].tolist(),
            "plot": f"data:image/png;base64,{b64}",
            "insight": insight,
        }
    )


@app.route("/api/report", methods=["GET"])
def get_report():
    """
    Endpoint GET /api/report
    Gera e retorna um relatório PDF consolidado do dataset, incluindo insights salvos.
    """
    dataset_id = request.args.get("dataset_id")
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset não encontrado"}), 404
    df = pd.read_csv(path)
    summary = quick_summary(df)
    pdf_path = os.path.join(DATA_DIR, f"{dataset_id}_report.pdf")
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    flow = []
    flow.append(Paragraph(f"Relatório do Dataset {dataset_id}", styles["Title"]))
    flow.append(Spacer(1, 12))
    flow.append(
        Paragraph(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}", styles["Normal"])
    )
    flow.append(Paragraph("Esquema:", styles["Heading2"]))
    flow.append(Paragraph(str(summary["schema"]), styles["Code"]))
    # incluir insights salvos
    from eda_agent import DB_CONN

    cur = DB_CONN.cursor()
    rows = cur.execute(
        "SELECT text, important FROM insights WHERE dataset_id=?", (dataset_id,)
    ).fetchall()
    flow.append(Paragraph("Insights:", styles["Heading2"]))
    for r in rows:
        txt = "⭐ " + r[0] if r[1] else r[0]
        flow.append(Paragraph(txt, styles["Normal"]))
    doc.build(flow)
    return send_file(pdf_path, as_attachment=True)


# --- Endpoints de insights ---
@app.route("/api/insights", methods=["GET"])
def get_insights():
    """
    Endpoint GET /api/insights
    Retorna todos os insights salvos para o dataset informado.
    """
    dataset_id = request.args.get("dataset_id")
    from eda_agent import DB_CONN

    cur = DB_CONN.cursor()
    rows = cur.execute(
        "SELECT insight_id, text, important, created_at FROM insights WHERE dataset_id=?",
        (dataset_id,),
    ).fetchall()
    insights = [
        {"id": r[0], "text": r[1], "important": bool(r[2]), "created_at": r[3]}
        for r in rows
    ]
    return jsonify(insights)


@app.route("/api/insights/mark", methods=["POST"])
def mark_insight():
    """
    Endpoint POST /api/insights/mark
    Marca um insight como importante no banco de dados.
    """
    data = request.get_json()
    iid = data.get("insight_id")
    from eda_agent import DB_CONN

    cur = DB_CONN.cursor()
    cur.execute("UPDATE insights SET important=1 WHERE insight_id=?", (iid,))
    DB_CONN.commit()
    return jsonify({"status": "ok"})


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """
    Endpoint GET /api/summary
    Retorna um resumo do dataset, incluindo esquema e gráficos básicos.
    """
    dataset_id = request.args.get("dataset_id")
    if not dataset_id:
        return jsonify({"error": "dataset_id é obrigatório"}), 400
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset_id não encontrado"}), 404
    df = pd.read_csv(path)
    summary = quick_summary(df)
    return (
        jsonify(
            {
                "dataset_id": dataset_id,
                "n_rows": df.shape[0],
                "n_cols": df.shape[1],
                "schema": summary["schema"],
                "plots": summary["plots"],
            }
        ),
        200,
    )


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """
    Endpoint POST /api/upload
    Realiza upload de um ou mais arquivos CSV, combina os dados e gera insights automáticos.
    """
    if "files" not in request.files:
        return (
            jsonify({"error": "Nenhum arquivo enviado, use 'files' no form-data"}),
            400,
        )
    files = request.files.getlist("files")
    dfs = []
    filenames = []
    for f in files:
        try:
            content = f.read()
            df = load_csv_bytes(content)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            filenames.append(f.filename)
        except Exception as e:
            return jsonify({"error": f"Falha ao ler {f.filename}: {str(e)}"}), 400

    if len(dfs) == 1:
        df_combined = dfs[0]
    else:
        common = set(dfs[0].columns)
        for d in dfs[1:]:
            common = common.intersection(set(d.columns))
        if common:
            key = list(common)[0]
            df_combined = dfs[0]
            for d in dfs[1:]:
                df_combined = df_combined.merge(d, on=key, how="outer")
        else:
            df_combined = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    dataset_id = str(uuid.uuid4())
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    df_combined.to_csv(path, index=False)

    save_dataset_metadata(dataset_id, ",".join(filenames), df_combined, path)

    summary = quick_summary(df_combined)

    # --- Geração automática de insights ---
    def generate_basic_insights(df, schema):
        """
        Gera insights automáticos simples a partir do DataFrame e do esquema inferido.
        """
        insights = []
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # Tendências
        for col in numeric_cols[:3]:
            mean_val = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            insights.append(f"A média da coluna '{col}' é {mean_val:.2f}.")
            insights.append(
                f"O valor mínimo de '{col}' é {min_val:.2f} e o máximo é {max_val:.2f}."
            )
            # Distribuição desbalanceada
            if abs(mean_val - min_val) < 0.1 * (max_val - min_val):
                insights.append(f"A coluna '{col}' possui distribuição desbalanceada.")
        # Riscos/limitações
        for col, info in schema.items():
            if info.get("missing", 0) > 0:
                insights.append(
                    f"A coluna '{col}' possui {info['missing']} valores ausentes."
                )
            if info.get("outliers", 0) > 0:
                insights.append(
                    f"A coluna '{col}' possui {info['outliers']} outliers detectados."
                )
        if not insights:
            insights.append(
                "Não foi possível gerar insights automáticos para este dataset."
            )
        return insights

    auto_insights = generate_basic_insights(df_combined, summary["schema"])
    from eda_agent import DB_CONN

    cur = DB_CONN.cursor()
    for text in auto_insights:
        cur.execute(
            "INSERT INTO insights VALUES (?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                dataset_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                text,
                0,
            ),
        )
    DB_CONN.commit()

    response = {
        "dataset_id": dataset_id,
        "n_rows": df_combined.shape[0],
        "n_cols": df_combined.shape[1],
        "schema": summary["schema"],
        "plots": summary["plots"],
    }
    return jsonify(response), 200


if __name__ == "__main__":
    print("Rodando Flask app: python agente_mvp.py")
    app.run(host="0.0.0.0", port=8000)


def try_answer_with_pandas(df, question: str):
    """
    Tenta responder perguntas simples sobre o DataFrame usando expressões regulares e Pandas.
    Retorna resposta, fonte e coluna, ou None se não encontrar correspondência.
    """
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
                if op == "mean":
                    return float(df[col].mean()), "pandas", col
                if op == "max":
                    return float(df[col].max()), "pandas", col
                if op == "min":
                    return float(df[col].min()), "pandas", col
                if op == "sum":
                    return float(df[col].sum()), "pandas", col
                if op == "count":
                    return int(df[col].count()), "pandas", col
    return None, None, None


def plot_histogram(df, col):
    """
    Gera um histograma em base64 para a coluna informada do DataFrame.
    """
    fig, ax = plt.subplots()
    df[col].hist(ax=ax)
    ax.set_title(f"Histograma de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequência")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


@app.route("/api/query", methods=["POST"])
def query():
    """
    Endpoint POST /api/query
    Responde perguntas em linguagem natural sobre o dataset, usando Pandas ou LLM.
    """
    data = request.get_json()
    if not data or "dataset_id" not in data or "question" not in data:
        return jsonify({"error": "Campos dataset_id e question são obrigatórios"}), 400

    dataset_id = data["dataset_id"]
    question = data["question"]
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "dataset_id não encontrado"}), 404
    df = pd.read_csv(path)

    # 1. Tentar responder com pandas
    ans, source, col = try_answer_with_pandas(df, question)
    if ans is not None:
        answer = f"A resposta para sua pergunta é: {ans}"
        plots = {}
        if col:
            try:
                plots[f"hist_{col}"] = plot_histogram(df, col)
            except Exception:
                pass
        save_query(dataset_id, question, answer, answer, source)
        return jsonify({"answer": answer, "source": source, "plots": plots})

    # 2. Se não deu match → usar LLM
    schema = infer_schema(df)
    sample = df.sample(min(20, len(df))).to_dict(orient="records")
    stats = df.describe(include="all").to_dict()
    context = {"schema": schema, "sample": sample, "stats": stats}
    prompt = f"""
    Você é um analista de dados. Responda a pergunta do usuário com base nos dados abaixo.\nPergunta: {question}\nContexto: {json.dumps(context)[:4000]}
    """
    try:
        llm_answer = call_gemini(prompt)
        save_query(dataset_id, question, llm_answer, llm_answer, "llm")
        return jsonify({"answer": llm_answer, "source": "llm", "plots": {}})
    except Exception as e:
        return jsonify({"error": f"Falha ao chamar LLM: {str(e)}"}), 500
