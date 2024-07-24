import os
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Constants
REPO_DIR = Path(os.path.dirname(__file__))
LOGS_DIR = REPO_DIR / "logs"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_FILE = LOGS_DIR / f"API_{timestamp}.log"
PDF_LIST = r"data\pdf_list.csv"
DB_FILE = r"data\database.db"

# Initialize logging directory
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True)
    print(f"Created {LOGS_DIR} directory.")

# Initialize logging file
logging.basicConfig(
    filename=LOGS_FILE,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Initialize FastAPI
app = FastAPI()

class AnalysisRequest(BaseModel):
    analysis_type: str
    parameters: dict = {}

@app.get("/api/")
def read_root():
    return {"Message": "Welcome to Project OCPM API! See README.md for more information."}

@app.post("/api/analysis/")
async def analyze(request: AnalysisRequest):
    try:
        data = pd.read_csv(PDF_LIST)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF list file not found. Contact administrator.")
    
    if request.analysis_type == "general":
        return general_analysis(data, request.parameters)
    elif request.analysis_type == "word_frequency":
        return word_frequency_analysis(data, request.parameters)
    elif request.analysis_type == "language_distribution":
        return language_distribution_analysis(data, request.parameters)
    elif request.analysis_type == "knowledge_type":
        return knowledge_type_analysis(data, request.parameters)
    elif request.analysis_type == "topic":
        return topic_analysis(data, request.parameters)
    elif request.analysis_type == "sentiment":
        return sentiment_analysis(data, request.parameters)
    else:
        raise HTTPException(status_code=404, detail="Analysis type not found. Contact administrator.")
    
def general_analysis(data, parameters):
    