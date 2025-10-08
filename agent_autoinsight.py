"""
agent_autoinsight.py

Gera insights automáticos para um dataset (heurísticas + LLM quando disponível)
Usa funções de eda_agent para estatísticas e plots.
"""
import json
import uuid
import time

from eda_agent import quick_summary, detect_outliers_iqr, histogram_plot, correlation_heatmap, DB_CONN
try:
    from call_gemini import call_gemini
except Exception:
    def call_gemini(prompt: str) -> str:
        return f"[Stub LLM] (autoinsight) {prompt[:300]}..."

def generate_insights(dataset_id: str, df):
    """
    Gera uma lista de insights (strings) e plots (dict col->base64)
    Salva insights no DB (tabela insights).
    Retorna {"insights": [...], "plots": {...}}
    """
    summary = quick_summary(df)
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "if"]
    insights = []
    plots = {}

    # 1) Estatísticas básicas para up to 3 numeric cols
    for col in numeric_cols[:3]:
        colinfo = summary["schema"].get(col, {})
        mean = colinfo.get("mean")
        median = colinfo.get("median")
        std = colinfo.get("std")
        n_missing = colinfo.get("missing", 0)
        insights.append(f"Coluna '{col}': média={mean}, mediana={median}, desvio padrão={std}, missing={n_missing}.")
        try:
            plots[f"hist_{col}"] = histogram_plot(df, col)
        except Exception:
            pass

    # 2) Outliers info (IQR)
    for col in numeric_cols[:3]:
        try:
            oi = detect_outliers_iqr(df[col].dropna())
            if oi["count"] > 0:
                insights.append(f"A coluna '{col}' possui {oi['count']} outliers (limites {oi['lower']:.2f} / {oi['upper']:.2f}).")
        except Exception:
            pass

    # 3) Correlação: top pair
    if len(numeric_cols) >= 2:
        try:
            _, corr_dict = correlation_heatmap(df, numeric_cols[:6])
            # find max abs corr pair
            import numpy as np
            import pandas as pd
            corr_df = pd.DataFrame(corr_dict)
            abs_corr = corr_df.abs()
            # zero diagonal and lower triangle
            arr = abs_corr.values.copy()
            arr[np.tril_indices_from(arr)] = 0
            idx = arr.argmax()
            i, j = divmod(idx, arr.shape[0])
            maxval = arr[i, j]
            if maxval > 0:
                colnames = list(corr_df.columns)
                insights.append(f"As colunas '{colnames[i]}' e '{colnames[j]}' têm correlação absoluta alta (r={maxval:.2f}).")
        except Exception:
            pass

    # 4) Narrativa via LLM (resumo executivo)
    try:
        prompt = f"Você é um analista de dados. Resuma em até 5 pontos acionáveis os resultados a seguir e indique 2 riscos/limitações.\nResumo estatístico: {json.dumps(summary['schema'])}\nInsights detectados por heurística: {insights[:6]}\n"
        llm_text = call_gemini(prompt)
        insights.insert(0, f"Narrativa LLM: {llm_text}")
    except Exception as e:
        insights.insert(0, f"[LLM-fallback] não foi possível gerar narrativa: {str(e)[:200]}")

    # 5) Persistir insights no DB
    try:
        cur = DB_CONN.cursor()
        for text in insights:
            cur.execute("INSERT INTO insights VALUES (?,?,?,?,?)", (
                str(uuid.uuid4()), dataset_id, time.strftime("%Y-%m-%d %H:%M:%S"), text, 0
            ))
        DB_CONN.commit()
    except Exception:
        # non-fatal; continue
        pass

    return {"insights": insights, "plots": plots, "schema": summary["schema"]}
