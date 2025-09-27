# streamlit frontend for the EDA agent MVP
import streamlit as st
import requests
import os

API_BASE = os.environ.get("EDA_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="EDA Agent MVP", layout="wide")
st.title("Agent E.D.A. — MVP")

st.sidebar.header("Envio de CSV")
uploaded = st.sidebar.file_uploader("Envie 1 ou mais CSVs", accept_multiple_files=True, type=['csv'])
if uploaded:
    files = uploaded
    with st.spinner("Enviando arquivos..."):
        files_payload = [('files', (f.name, f.getvalue(), 'text/csv')) for f in files]
        try:
            resp = requests.post(f"{API_BASE}/api/upload", files=files_payload, timeout=120)
            if resp.status_code == 200:
                payload = resp.json()
                st.success("Upload concluído")
                st.session_state['dataset_id'] = payload['dataset_id']
                st.subheader("Resumo Inicial")
                st.markdown(f"- Linhas: **{payload['n_rows']}**  \n- Colunas: **{payload['n_cols']}**")
                st.markdown("**Esquema (amostra)**")
                st.json(payload['schema'])
                if payload.get('plots'):
                    st.markdown("**Gráficos gerados automaticamente**")
                    for name, b64 in payload['plots'].items():
                        st.image(b64, use_column_width=True)
            else:
                st.error(f"Erro no upload: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Erro: {e}")

if 'dataset_id' in st.session_state:
    ds = st.session_state['dataset_id']
    st.sidebar.markdown(f"Dataset carregado: `{ds}`")
    if st.sidebar.button("Recarregar resumo"):
        try:
            resp = requests.get(f"{API_BASE}/api/summary", params={"dataset_id": ds}, timeout=30)
            if resp.status_code == 200:
                payload = resp.json()
                st.subheader("Resumo Atualizado")
                st.markdown(f"- Linhas: **{payload['n_rows']}**  \n- Colunas: **{payload['n_cols']}**")
                st.json(payload['schema'])
                if payload.get('plots'):
                    for name, b64 in payload['plots'].items():
                        st.image(b64, use_column_width=True)
            else:
                st.error("Falha ao obter resumo")
        except Exception as e:
            st.error(str(e))
    st.markdown("---")
    st.markdown("Faça perguntas sobre o dataset (em breve funcionalidade de query)")
    question = st.text_input("Pergunta (ex.: Qual a média da coluna 'valor'?)")
    if st.button("Enviar pergunta"):
        st.info("Funcionalidade de perguntas será integrada na próxima sprint.")