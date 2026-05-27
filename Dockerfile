FROM python:3.11-slim

WORKDIR /app

# install system deps spacy needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download spacy model
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8501

RUN python -c "from transformers import pipeline; pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]