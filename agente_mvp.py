from flask import Flask, request, jsonify
import os, io, uuid, time
import pandas as pd
from eda_agent import load_csv_bytes, quick_summary, save_dataset_metadata
app = Flask(__name__)

DATA_DIR = os.environ.get("EDA_DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if 'files' not in request.files:
        return jsonify({"error":"Nenhum arquivo enviado, use 'files' no form-data"}), 400
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
                df_combined = df_combined.merge(d, on=key, how='outer')
        else:
            df_combined = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    dataset_id = str(uuid.uuid4())
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    df_combined.to_csv(path, index=False)

    save_dataset_metadata(dataset_id, ",".join(filenames), df_combined, path)

    summary = quick_summary(df_combined)
    response = {"dataset_id": dataset_id, "n_rows": df_combined.shape[0], "n_cols": df_combined.shape[1],
                "schema": summary['schema'], "plots": summary['plots']}
    return jsonify(response), 200

@app.route("/api/summary", methods=["GET"])
def get_summary():
    dataset_id = request.args.get("dataset_id")
    if not dataset_id:
        return jsonify({"error":"dataset_id é obrigatório"}), 400
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return jsonify({"error":"dataset_id não encontrado"}), 404
    df = pd.read_csv(path)
    summary = quick_summary(df)
    return jsonify({"dataset_id": dataset_id, "n_rows": df.shape[0], "n_cols": df.shape[1],
                    "schema": summary['schema'], "plots": summary['plots']}), 200

if __name__ == "__main__":
    print("Rodando Flask app: python agente_mvp.py")
    app.run(host="0.0.0.0", port=8000)