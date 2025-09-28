# streamlit frontend for the EDA agent MVP
import streamlit as st
import requests
import os

API_BASE = os.environ.get("EDA_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="EDA Agent MVP", layout="wide")
st.title("Agent E.D.A. — MVP")

st.sidebar.header("Envio de CSV")
uploaded = st.sidebar.file_uploader(
    "Envie 1 ou mais CSVs", accept_multiple_files=True, type=["csv"]
)
if uploaded:
    files = uploaded
    with st.spinner("Enviando arquivos..."):
        files_payload = [("files", (f.name, f.getvalue(), "text/csv")) for f in files]
        try:
            resp = requests.post(
                f"{API_BASE}/api/upload", files=files_payload, timeout=120
            )
            if resp.status_code == 200:
                payload = resp.json()
                st.success("Upload concluído")
                st.session_state["dataset_id"] = payload["dataset_id"]
                st.subheader("Resumo Inicial")
                st.markdown(
                    f"- Linhas: **{payload['n_rows']}**  \n- Colunas: **{payload['n_cols']}**"
                )
                st.markdown("**Esquema (amostra)**")
                st.json(payload["schema"])
                if payload.get("plots"):
                    st.markdown("**Gráficos gerados automaticamente**")
                    for name, b64 in payload["plots"].items():
                        st.image(b64, use_column_width=True)
            else:
                st.error(f"Erro no upload: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Erro: {e}")

if "dataset_id" in st.session_state:
    ds = st.session_state["dataset_id"]
    st.markdown("---")
    tabs = st.tabs(["Chat", "Outliers", "Correlação", "Clusters", "Exportar PDF"])

    # Chat
    with tabs[0]:
        st.subheader("Chat com o dataset")
        question = st.text_input("Pergunte em linguagem natural:")
        if st.button("Enviar", key="chat_send"):
            if not question.strip():
                st.warning("Digite uma pergunta.")
            else:
                with st.spinner("Consultando..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/query",
                            json={"dataset_id": ds, "question": question},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            payload = resp.json()
                            st.success(
                                f"Resposta ({payload.get('source','?')}): {payload.get('answer','[sem resposta]')}"
                            )
                            if payload.get("plots"):
                                for name, b64 in payload["plots"].items():
                                    st.markdown(f"**{name}**")
                                    st.image(b64, use_column_width=True)
                        else:
                            st.error(f"Erro: {resp.text}")
                    except Exception as e:
                        st.error(str(e))

    # Outliers
    with tabs[1]:
        st.subheader("Detecção de Outliers")
        try:
            resp = requests.get(
                f"{API_BASE}/api/summary", params={"dataset_id": ds}, timeout=30
            )
            if resp.status_code == 200:
                payload = resp.json()
                num_cols = [
                    c
                    for c, v in payload["schema"].items()
                    if v.get("dtype", "").startswith("float")
                    or v.get("dtype", "").startswith("int")
                ]
                col = st.selectbox("Selecione a coluna numérica:", num_cols)
                if st.button("Detectar Outliers", key="outliers_btn"):
                    try:
                        resp2 = requests.get(
                            f"{API_BASE}/api/outliers",
                            params={"dataset_id": ds, "col": col},
                            timeout=30,
                        )
                        if resp2.status_code == 200:
                            out = resp2.json()
                            st.markdown(f"**Boxplot:**")
                            st.image(out["plot"], use_column_width=True)
                            st.markdown(f"**Estatísticas:**")
                            st.json(out["stats"])
                        else:
                            st.error(resp2.text)
                    except Exception as e:
                        st.error(str(e))
        except Exception as e:
            st.error(str(e))

    # Correlação
    with tabs[2]:
        st.subheader("Correlação entre colunas")
        try:
            resp = requests.get(
                f"{API_BASE}/api/correlation", params={"dataset_id": ds}, timeout=30
            )
            if resp.status_code == 200:
                payload = resp.json()
                st.markdown("**Heatmap de Correlação:**")
                st.image(payload["plot"], use_column_width=True)
                st.markdown("**Matriz de Correlação:**")
                st.json(payload["corr"])
                st.info(payload.get("insight", ""))
            else:
                st.error(resp.text)
        except Exception as e:
            st.error(str(e))

    # Clusters
    with tabs[3]:
        st.subheader("Clustering KMeans")
        try:
            resp = requests.get(
                f"{API_BASE}/api/summary", params={"dataset_id": ds}, timeout=30
            )
            if resp.status_code == 200:
                payload = resp.json()
                num_cols = [
                    c
                    for c, v in payload["schema"].items()
                    if v.get("dtype", "").startswith("float")
                    or v.get("dtype", "").startswith("int")
                ]
                xcol = st.selectbox("Coluna X:", num_cols, key="cluster_x")
                ycol = st.selectbox("Coluna Y:", num_cols, key="cluster_y")
                k = st.slider("Número de clusters:", 2, 5, 3)
                if st.button("Rodar Clustering", key="cluster_btn"):
                    try:
                        resp2 = requests.get(
                            f"{API_BASE}/api/clusters",
                            params={"dataset_id": ds, "cols": f"{xcol},{ycol}", "k": k},
                            timeout=30,
                        )
                        if resp2.status_code == 200:
                            cl = resp2.json()
                            st.markdown("**Scatter plot dos clusters:**")
                            st.image(cl["plot"], use_column_width=True)
                            st.info(cl.get("insight", ""))
                        else:
                            st.error(resp2.text)
                    except Exception as e:
                        st.error(str(e))
        except Exception as e:
            st.error(str(e))

    # Exportação PDF
    with tabs[4]:
        st.subheader("Exportar relatório PDF")
        if st.button("Exportar PDF", key="pdf_btn"):
            with st.spinner("Gerando relatório..."):
                try:
                    resp = requests.get(
                        f"{API_BASE}/api/report", params={"dataset_id": ds}, timeout=60
                    )
                    if resp.status_code == 200:
                        pdf_bytes = resp.content
                        st.download_button(
                            "Baixar relatório PDF",
                            data=pdf_bytes,
                            file_name=f"{ds}_report.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.error("Erro ao gerar PDF")
                except Exception as e:
                    st.error(str(e))
    ds = st.session_state["dataset_id"]
    st.sidebar.markdown(f"Dataset carregado: `{ds}`")
    # Painel de insights
    st.sidebar.subheader("💡 Memória do agente")
    try:
        resp = requests.get(
            f"{API_BASE}/api/insights", params={"dataset_id": ds}, timeout=20
        )
        if resp.status_code == 200:
            insights = resp.json()
            for ins in insights:
                if ins["important"]:
                    st.sidebar.markdown(
                        f"<div style='background-color:#fffbe6;padding:4px;border-radius:4px;margin-bottom:2px'>✅ <b>{ins['text']}</b></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.sidebar.write(f"- {ins['text']}")
                    if st.sidebar.button(f"Marcar importante", key=ins["id"]):
                        requests.post(
                            f"{API_BASE}/api/insights/mark",
                            json={"insight_id": ins["id"]},
                        )
                        st.sidebar.experimental_rerun()
        else:
            st.sidebar.error("Falha ao buscar insights")
    except Exception as e:
        st.sidebar.error(f"Erro ao buscar insights: {e}")

    if st.sidebar.button("Recarregar resumo"):
        try:
            resp = requests.get(
                f"{API_BASE}/api/summary", params={"dataset_id": ds}, timeout=30
            )
            if resp.status_code == 200:
                payload = resp.json()
                st.subheader("Resumo Atualizado")
                st.markdown(
                    f"- Linhas: **{payload['n_rows']}**  \n- Colunas: **{payload['n_cols']}**"
                )
                st.json(payload["schema"])
                if payload.get("plots"):
                    for name, b64 in payload["plots"].items():
                        st.image(b64, use_column_width=True)
            else:
                st.error("Falha ao obter resumo")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("Chat com o dataset")
    question = st.text_input("Pergunte em linguagem natural:")
    if st.button("Enviar"):
        if not question.strip():
            st.warning("Digite uma pergunta.")
        else:
            with st.spinner("Consultando..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/query",
                        json={"dataset_id": ds, "question": question},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        payload = resp.json()
                        st.success(
                            f"Resposta ({payload.get('source','?')}): {payload.get('answer','[sem resposta]')}"
                        )
                        if payload.get("plots"):
                            for name, b64 in payload["plots"].items():
                                st.markdown(f"**{name}**")
                                st.image(b64, use_column_width=True)
                    else:
                        st.error(f"Erro: {resp.text}")
                except Exception as e:
                    st.error(str(e))
