# IMDb Movie Genre Classification 
# Modeling Pipeline and App

## Overview

This project aims to use movie plot descriptions to accurately determine movie genre using four different multiclassification models. This repo contains an end-to-end ML pipeline that is containerized with Docker, alllowing users to build and run the model pipeline and deploy an interactive Streamlit app to enter a movie summary and receive model genre predictions. 

The pipeline does the following: 
- Reads in dataset from Kaggle on IMDb movie genres; cleans and processes the data
- Trains four different models to predict movie genre: Naive Bayes, Logistic Regression, LSTM, and transformer
- Evaluates model performance on test data and saves metrics like accuracy and confusion matrices and word cloud visualiztaions for each model
- Deploys interactive Streamlit web app that takes a user-provided movie summary (real or fake), cleans text, and returns genre prediction probabilities from each model

## Repository Structures

```text
NLP-MOVIE-CLASSIFICATION/
├── app/                        # Streamlit deployment application
│   ├── app.py                  # Main Streamlit application file
│   ├── Dockerfile.app          # Docker configuration for app deployment
│   ├── requirements_streamlit.txt  # Dependencies specific to Streamlit app
│   └── streamlit_pipeline.py   # Inference/prediction pipeline helper for Streamlit
│
├── artifacts/                  
│   ├── metrics/                # Stored evaluation reports and confusion matrices 
│   ├── models/                 # Saved model files
│
├── config/                     
│    ├── config.yaml/           # Contains data, models, paths, hyperparameters, and files for pipeline
│
├── src/                        
│   ├── __init__.py             # Makes src a Python packagable directory
│   ├── ingest.py               # Data loading and ingestion 
│   ├── logistic_regression.py  # trains and pickles logistic regression model 
│   ├── lstm.py                 # trains and pickles LSTM model 
│   ├── naive_bayes.py          # trains and pickles Naive Bayes model
│   ├── process.py              # Cleans and tokenizes data; creates summarization tool
│   ├── transformer.py          # trains and pickles transformer model
│   └── utils.py                # Helper functions
│
├── .gitignore                  # Specifies files to ignore in Git
├── .dockerignore               # Specifies files to ignore when building Docker image
├── docker-compose.yml          # Multi-container Docker orchestration configuration
├── Dockerfile.pipeline         # Docker configuration for running the ML pipeline
├── main.py                     # Main execution script to run pipeline
├── Project_NLP.ipynb           # Notebook containing data processing and modeling, and additional EDA including misclassification
├── README.md                   # Project documentation and instructions to run 
└── requirements.txt            # Project dependencies
```

## How to Run

Follow the instructions below to execute the pipeline on your local machine. 

### 1. Clone repository to local machine and set working directory to repository 

### 2. Optional: Create a new virtual or conda environment if desired (not necessary)

### 3. Ensure Docker is installed and open 

### 4. Update config/config.yaml

Update "models" section of config.yaml with "True" for any models you want to run the pipeline for and "False" for any models you don't. Update any other config variables, like model parameters, if desired. 

### 5. Build Docker containers for pipeline and app 

Run the following commands in your terminal to build and execute Docker environments for the ML pipeline and Streamlit app:  

```bash
docker-compose build pipeline
``` 
```bash
docker-compose run pipeline
``` 
```bash
docker-compose up app 
``` 
### 6. Open Streamlit app

Copy the URL that appears in terminal or open browser to http://localhost:8501 to access the interactive Streamlit app.

### 7. Use Streamlit

- Enter a movie description of choice (real or fake) into the given textbox and hit command-return
- Select the models that you want to use to predict the genre 
- Click "predict genre" to see cleaned text and genre prediction probabilities per model

## Authors

### Arielle Weinstein, Aru Fatehpuria, Blake Wood, Jill Cusick, and Pippa Hodgkins
 