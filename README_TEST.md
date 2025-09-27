# EDA Agent MVP - Test Instructions

## Requirements
- Python 3.10+
- Install required packages:
  pip install flask pandas matplotlib scikit-learn streamlit requests

## Run backend (Flask)
$ python /mnt/data/agente_mvp.py

This will start the API at http://0.0.0.0:8000

## Run frontend (Streamlit)
In another terminal:
$ streamlit run /mnt/data/front_streamlit.py

Open the Streamlit UI and upload CSV files.

## Test with curl
Upload a CSV:
curl -X POST "http://localhost:8000/api/upload" -F "files=@/path/to/your.csv"

Get summary:
curl "http://localhost:8000/api/summary?dataset_id=<dataset_id_returned_from_upload>"

## Notes
- Uploaded CSVs are stored in ./data as <dataset_id>.csv
- A simple SQLite DB memory.db stores dataset metadata and queries.
- This is the Sprint A MVP: upload -> schema inference -> basic plots as base64 images -> summary endpoint.
- Next steps: implement /api/query (NLP), outlier endpoints, clustering, and persistent insight saving.