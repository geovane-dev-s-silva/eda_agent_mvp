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

## Sobre:
Agentes Autônomos – Relatório da Atividade
1. Framework Escolhida
Para o desenvolvimento do EDA Agent MVP, foi escolhida uma arquitetura baseada em Docker Compose com os seguintes componentes principais:

Backend: Flask (Python) com endpoints RESTful

Frontend: Streamlit para interface web interativa

Banco de Dados: SQLite para persistência de metadados e histórico

Orquestração: Docker Compose para containerização

Processamento de Dados: Pandas, NumPy, Scikit-learn

Visualização: Matplotlib

IA Generativa: Google Gemini API (opcional)

2. Estrutura da Solução
Arquitetura Modular
A solução foi estruturada em módulos especializados:

text
EDA Agent MVP/
├── agente_mvp.py (Backend principal - Flask)
├── front_streamlit.py (Frontend - Streamlit)
├── eda_agent.py (Núcleo de análise exploratória)
├── agent_memory.py (Gerenciamento de memória)
├── agent_autoinsight.py (Geração de insights automáticos)
├── call_gemini.py (Integração com LLM)
├── requirements.txt
├── docker-compose.yml
└── README.md

Fluxo de Processamento
Upload de Dados: Usuário envia arquivos CSV via interface Streamlit

Processamento Inicial:

Combinação de múltiplos arquivos (quando aplicável)

Inferência automática de schema e estatísticas

Geração de gráficos básicos

Análise Interativa:

Chat em linguagem natural

Detecção de outliers

Análise de correlação

Clustering K-means

Persistência:

Metadados no SQLite

Histórico de perguntas e respostas

Insights gerados automaticamente

Componentes Principais
Backend (Flask):

/api/upload - Processamento de uploads

/api/query - Processamento de perguntas em NL

/api/outliers - Detecção de outliers

/api/correlation - Análise de correlação

/api/clusters - Clusterização K-means

/api/report - Geração de relatório PDF

Frontend (Streamlit):

Interface tabular para diferentes funcionalidades

Visualização interativa de gráficos

Gerenciamento de estado da sessão

Sidebar com insights e memória

3. Códigos Fonte

agente_mvp.py

front_streamlit.py

eda_agent.py

agent_memory.py

agent_autoinsight.py

call_gemini.py

requirements.txt

docker-compose.yml

Data de Geração: 09 de Outubro de 2025
Versão do Agente: MVP 1.0
Capacidades: Análise exploratória, processamento de linguagem natural, detecção de padrões, geração de insights automáticos