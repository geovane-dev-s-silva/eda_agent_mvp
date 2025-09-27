import os
import io
import time
import json
import base64
import uuid
import sqlite3
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DB_PATH = os.environ.get("EDA_DB_PATH", "db/memory.db")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,
        name TEXT,
        uploaded_at TEXT,
        n_rows INTEGER,
        n_cols INTEGER,
        filepath TEXT,
        schema_json TEXT
    )"""
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS queries (
        query_id TEXT PRIMARY KEY,
        dataset_id TEXT,
        question TEXT,
        response_summary TEXT,
        raw_response TEXT,
        created_at TEXT,
        source TEXT
    )"""
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS insights (
        insight_id TEXT PRIMARY KEY,
        dataset_id TEXT,
        created_at TEXT,
        text TEXT,
        important INTEGER DEFAULT 0
    )"""
    )
    conn.commit()
    return conn


DB_CONN = init_db()


def save_dataset_metadata(dataset_id, name, df, filepath):
    cur = DB_CONN.cursor()
    schema = infer_schema(df)
    cur.execute(
        "REPLACE INTO datasets (dataset_id,name,uploaded_at,n_rows,n_cols,filepath,schema_json) VALUES (?,?,?,?,?,?,?)",
        (
            dataset_id,
            name,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            df.shape[0],
            df.shape[1],
            filepath,
            json.dumps(schema),
        ),
    )
    DB_CONN.commit()


def save_query(dataset_id, question, response, raw, source):
    cur = DB_CONN.cursor()
    qid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO queries VALUES (?,?,?,?,?,?,?)",
        (
            qid,
            dataset_id,
            question,
            response[:2000],
            raw[:2000],
            time.strftime("%Y-%m-%d %H:%M:%S"),
            source,
        ),
    )
    DB_CONN.commit()
    return qid


def infer_schema(df):
    schema = {}
    for col in df.columns:
        colseries = df[col]
        dtype = str(colseries.dtype)
        n_missing = int(colseries.isna().sum())
        n_unique = int(colseries.nunique(dropna=True))
        samples = []
        try:
            samples = (
                colseries.dropna()
                .sample(min(3, max(1, colseries.dropna().shape[0])))
                .astype(str)
                .tolist()
            )
        except Exception:
            samples = colseries.dropna().astype(str).tolist()[:3]
        colinfo = {
            "dtype": dtype,
            "missing": n_missing,
            "unique": n_unique,
            "sample": samples,
        }
        if pd.api.types.is_numeric_dtype(colseries):
            colinfo.update(
                {
                    "min": None if colseries.dropna().empty else float(colseries.min()),
                    "max": None if colseries.dropna().empty else float(colseries.max()),
                    "mean": (
                        None if colseries.dropna().empty else float(colseries.mean())
                    ),
                    "median": (
                        None if colseries.dropna().empty else float(colseries.median())
                    ),
                    "std": None if colseries.dropna().empty else float(colseries.std()),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(colseries):
            colinfo.update(
                {
                    "min": None if colseries.dropna().empty else str(colseries.min()),
                    "max": None if colseries.dropna().empty else str(colseries.max()),
                }
            )
        schema[col] = colinfo
    return schema


def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def histogram_plot(df, col):
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        df[col].dropna().astype(float).plot(kind="hist", bins=30, ax=ax)
    except Exception:
        df[col].dropna().plot(kind="hist", bins=30, ax=ax)
    ax.set_title(f"Histograma: {col}")
    return plot_to_base64(fig)


def boxplot_plot(df, col):
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        df[col].dropna().astype(float).plot(kind="box", ax=ax)
    except Exception:
        df[col].dropna().plot(kind="box", ax=ax)
    ax.set_title(f"Boxplot: {col}")
    return plot_to_base64(fig)


def scatter_plot(df, x, y):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df[x], df[y], alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    return plot_to_base64(fig)


def correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)
    ax.set_title("Matriz de Correlação")
    return plot_to_base64(fig), corr.to_dict()


def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "count": int(outliers.shape[0]),
        "lower": float(lower),
        "upper": float(upper),
    }


def load_csv_bytes(content_bytes):
    # Try to infer delimiter, fallback to comma
    try:
        sample = content_bytes.decode("utf-8", errors="ignore").splitlines()[:10]
        # basic heuristic: semicolon or comma
        if any(";" in line for line in sample):
            sep = ";"
        elif any("\t" in line for line in sample):
            sep = "\t"
        else:
            sep = ","
        df = pd.read_csv(io.BytesIO(content_bytes), sep=sep, engine="python")
        return df
    except Exception:
        return pd.read_csv(io.BytesIO(content_bytes))


def quick_summary(df):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    schema = infer_schema(df)
    plots = {}
    for col in numeric_cols[:3]:
        try:
            plots[f"hist_{col}"] = histogram_plot(df, col)
            plots[f"box_{col}"] = boxplot_plot(df, col)
        except Exception:
            continue
    corr_plot = None
    corr_json = {}
    if len(numeric_cols) >= 2:
        try:
            corr_plot, corr_json = correlation_heatmap(df, numeric_cols[:6])
            plots["corr_heatmap"] = corr_plot
        except Exception:
            pass
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "schema": schema,
        "plots": plots,
        "corr": corr_json,
    }
