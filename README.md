# EDA Agent MVP

Um agente de exploração de dados (EDA) automatizado, fácil de usar, para análise exploratória de datasets CSV.

### Como inserir sua chave de API Gemini (perguntas em linguagem natural)

Para que o agente responda perguntas em linguagem natural, é necessário obter uma chave de API Gemini (Google):

1. Crie uma conta e gere sua chave em: https://aistudio.google.com/app/apikey
2. No arquivo `docker-compose.yml`, localize a linha:
  ```yaml
  - GEMINI_API_KEY=your_google_gemini_api_key_here
  ```
3. Substitua `your_google_gemini_api_key_here` pela sua chave real, mantendo as aspas.
4. Salve o arquivo e rode o comando:
  ```
  docker-compose up --build
  ```

Pronto! Agora o agente poderá responder perguntas em linguagem natural usando a API Gemini.

## Principais Funcionalidades
- Upload de um ou mais arquivos CSV.
- Resumo automático do dataset (esquema, estatísticas, gráficos).
- Perguntas em linguagem natural sobre os dados (ex: "Qual a média da coluna valor?").
- Detecção de outliers, análise de correlação, clustering KMeans.
- Geração de relatório PDF consolidado.
- Memória de insights e histórico de perguntas.

## Como rodar 

**Pré-requisitos:**
- [Instale o Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Passos:**
1. Abra o terminal na pasta do projeto.
2. Execute:
   ```
   docker-compose up --build
   ```
3. Aguarde a mensagem de que os serviços "eda_backend" e "eda_frontend" estão rodando.
4. Acesse [http://localhost:8501] no navegador.
5. Envie seus arquivos CSV e explore as abas: Resumo, Chat, Outliers, Correlação, Relatório.

**Dica:** Não é necessário instalar Python ou pacotes manualmente!

## Fluxo de uso
1. Faça upload de um arquivos no formato CSV.
2. Veja o resumo automático e gráficos.
3. Use o chat para perguntar em português sobre os dados.
4. Explore outliers, correlações e clusters nas abas.
5. Baixe o relatório PDF com tudo consolidado.

## Estrutura técnica
- Backend: Flask (agente_mvp.py), SQLite, Pandas, Matplotlib, Scikit-learn, Reportlab.
- Frontend: Streamlit (front_streamlit.py).
- Orquestração: Docker Compose.

## Dúvidas comuns
- **Onde ficam meus dados?**
  - Os arquivos enviados ficam na pasta `data/`.
  - O banco de dados (insights, histórico) fica em `db/`.
- **Posso rodar sem Docker?**
  - Sim, mas Docker é recomendado para evitar problemas de dependências.
- **Como parar?**
  - Use `CTRL+C` no terminal ou `docker-compose down`.

---
Projeto MVP para exploração de dados sem complicação. Dúvidas? Abra uma issue ou peça ajuda!
