import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="EDA Multiagente", layout="wide")
API_BASE = os.getenv("EDA_API_BASE", "http://backend:8000")

st.title("🔍 Agente de Exploração de Dados (EDA)")
uploaded_files = st.file_uploader(
    "Envie um ou mais arquivos CSV", type=["csv"], accept_multiple_files=True
)

if uploaded_files:
    if st.button("Enviar para análise"):
        with st.spinner("Processando..."):
            files = [
                ("files", (f.name, f.getvalue(), "text/csv")) for f in uploaded_files
            ]
            r = requests.post(f"{API_BASE}/api/upload", files=files, timeout=180)
            if r.status_code == 200:
                data = r.json()
                st.session_state["dataset_id"] = data["dataset_id"]
                st.success("Upload concluído!")
            else:
                st.error(r.text)

if "dataset_id" in st.session_state:
    ds = st.session_state["dataset_id"]
    tabs = st.tabs(["Resumo", "Chat", "Outliers", "Correlação", "Relatório"])

    with tabs[0]:
        st.subheader("Resumo dos Dados")
        st.write(f"ID: {ds}")
        st.write("Esquema detectado:")
        resp = requests.get(f"{API_BASE}/api/report", params={"dataset_id": ds})
        if resp.status_code == 200:
            st.download_button(
                "📄 Baixar Relatório", resp.content, file_name="relatorio.pdf"
            )

    with tabs[1]:
        st.subheader("Chat com o Dataset")
        question = st.text_input("Faça uma pergunta:")
        if st.button("Enviar Pergunta"):
            r = requests.post(
                f"{API_BASE}/api/query", json={"dataset_id": ds, "question": question}
            )
            if r.status_code == 200:
                res = r.json()
                st.info(f"Resposta ({res['source']}): {res['answer']}")
            else:
                st.error(r.text)

    with tabs[2]:
        st.subheader("Detecção de Outliers")
        df = pd.read_csv(f"data/{ds}.csv")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            st.warning("Nenhuma coluna numérica encontrada.")
        else:
            col = st.selectbox("Selecione uma coluna:", num_cols)
            r = requests.get(
                f"{API_BASE}/api/outliers", params={"dataset_id": ds, "col": col}
            )
            if r.status_code == 200:
                data = r.json()
                st.json(data["stats"])
                st.image(data["plot"])

    with tabs[3]:
        st.subheader("Correlação entre variáveis")
        r = requests.get(f"{API_BASE}/api/correlation", params={"dataset_id": ds})
        if r.status_code == 200:
            st.image(r.json()["plot"])
