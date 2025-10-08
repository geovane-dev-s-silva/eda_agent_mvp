import streamlit as st
import requests
import os
import pandas as pd

API_BASE = os.getenv("EDA_API_BASE", "http://backend:8000")

st.set_page_config(page_title="EDA Multiagente", layout="wide")
st.title("🔍 Agente de Exploração de Dados (EDA) - v1.1")

uploaded_files = st.file_uploader("Envie um ou mais arquivos CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Enviar para análise"):
        with st.spinner("Processando..."):
            files = [("files", (f.name, f.getvalue(), "text/csv")) for f in uploaded_files]
            r = requests.post(f"{API_BASE}/api/upload", files=files, timeout=180)
            if r.status_code == 200:
                data = r.json()
                st.session_state["dataset_id"] = data["dataset_id"]
                st.success("Upload concluído!")
            else:
                st.error(r.text)

if "dataset_id" in st.session_state:
    ds = st.session_state["dataset_id"]
    tabs = st.tabs(["Resumo", "Insights", "Chat", "Outliers", "Correlação", "Relatório"])

    with tabs[0]:
        st.subheader("Resumo dos Dados")
        st.write(f"ID: {ds}")
        resp = requests.get(f"{API_BASE}/api/summary", params={"dataset_id": ds})
        if resp.status_code == 200:
            payload = resp.json()
            st.markdown(f"- Linhas: **{payload['n_rows']}**  \n- Colunas: **{payload['n_cols']}**")
            st.json(payload["schema"])
            if payload.get("plots"):
                for name, b64 in payload["plots"].items():
                    st.image(b64, use_column_width=True)

    with tabs[1]:
        st.subheader("💡 Insights Automáticos")
        try:
            resp = requests.get(f"{API_BASE}/api/insights", params={"dataset_id": ds}, timeout=20)
            if resp.status_code == 200:
                insights = resp.json()
                for ins in insights:
                    if ins["important"]:
                        st.markdown(f"**⭐ {ins['text']}**")
                    else:
                        st.write(f"- {ins['text']}")
            else:
                st.error("Falha ao buscar insights")
        except Exception as e:
            st.error(str(e))

    with tabs[2]:
        st.subheader("Chat com o Dataset (com memória)")
        question = st.text_input("Faça uma pergunta:")
        if st.button("Enviar Pergunta"):
            r = requests.post(f"{API_BASE}/api/query", json={"dataset_id": ds, "question": question}, timeout=60)
            if r.status_code == 200:
                res = r.json()
                st.info(f"Resposta ({res['source']}): {res['answer']}")
            else:
                st.error(r.text)

    with tabs[3]:
        st.subheader("Detecção de Outliers")
        resp = requests.get(f"{API_BASE}/api/summary", params={"dataset_id": ds})
        if resp.status_code == 200:
            payload = resp.json()
            num_cols = [c for c,v in payload["schema"].items() if v.get("dtype","").startswith("float") or v.get("dtype","").startswith("int")]
            if not num_cols:
                st.warning("Nenhuma coluna numérica encontrada.")
            else:
                col = st.selectbox("Selecione uma coluna:", num_cols)
                if st.button("Detectar Outliers"):
                    r = requests.get(f"{API_BASE}/api/outliers", params={"dataset_id": ds, "col": col})
                    if r.status_code == 200:
                        data = r.json()
                        st.json(data["stats"])
                        st.image(data["plot"])
                    else:
                        st.error(r.text)

    with tabs[4]:
        st.subheader("Correlação entre variáveis")
        r = requests.get(f"{API_BASE}/api/correlation", params={"dataset_id": ds})
        if r.status_code == 200:
            st.image(r.json()["plot"])
            st.json(r.json().get("corr", {}))

    with tabs[5]:
        st.subheader("Exportar Relatório PDF")
        if st.button("Exportar PDF"):
            resp = requests.get(f"{API_BASE}/api/report", params={"dataset_id": ds}, timeout=60)
            if resp.status_code == 200:
                st.download_button("📄 Baixar Relatório", resp.content, file_name=f"{ds}_report.pdf", mime="application/pdf")
            else:
                st.error("Erro ao gerar PDF")
