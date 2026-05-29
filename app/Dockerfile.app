FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements_streamlit.txt .
RUN pip install --no-cache-dir -r requirements_streamlit.txt
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
RUN python -m nltk.downloader stopwords punkt_tab wordnet

COPY artifacts/ artifacts/
COPY app/ app/

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]