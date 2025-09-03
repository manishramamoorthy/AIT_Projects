# Import Libraries
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional, Dict
import json
import time
import os

from data_preprocessing import DataProcessing
from metadata_extraction import MetaDataExtraction
from ai_refinement import AIPipeline
from asset_management import AssetManagement
from data_persistence import DataPersistence

load_dotenv()
# Initialize a APP
app = FastAPI(title="Data Optimization & Management Microservice")

# Initialize a config :
API_KEY = os.getenv('API_KEY')
API_KEY_NAME = os.getenv('API_KEY_NAME')
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

RATE_LIMIT = 5  # max requests per IP
TIME_WINDOW = 60  # seconds
request_log: Dict[str, list] = {}  # track requests per client
RESULTS_DIR = "."  # folder where result files are stored


# Authentication Process : verify the API key provided in the request header
def get_api_key(api_key: str = Depends(api_key_header)):
    """Check if the provided API key is valid; raise 403 error if not."""
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
    return api_key


# Rate Limit Process : limit the number of requests per client IP within a time window
def rate_limiter(request: Request):
    """Check request frequency for the client IP and block if the rate limit is exceeded."""
    client_ip = request.client.host
    now = time.time()

    if client_ip not in request_log:
        request_log[client_ip] = []
    request_log[client_ip] = [
        t for t in request_log[client_ip] if now - t < TIME_WINDOW
    ]

    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429, detail="Too Many Requests. Try again later."
        )

    request_log[client_ip].append(now)


# Pyndatic Model
class Review(BaseModel):
    rating: Optional[float] = None
    timestamp: Optional[str] = None  # ISO 8601 string
    text: Optional[str] = None


# Endpoint :  Run the full data optimization pipeline
@app.post("/optimize")
def process_pipeline(reviews: List[Review]):
    """Process a list of reviews through preprocessing, metadata extraction, AI refinement,
       asset management, and data persistence, then return the results and log info."""
    if not reviews:
        raise HTTPException(status_code=400, detail="No data provided")

    # Step 1: Data Preprocessing
    data_list = [review.dict() for review in reviews]
    processor = DataProcessing(data_list)
    processor.missing_values()
    processor.remove_stopwords_and_lemmatize()
    processor.save_to_json("cleaned_data.json")

    # Step 2: Metadata Extraction
    extractor = MetaDataExtraction(
        input_file="cleaned_data.json", output_file="meta_data.json"
    )
    extractor.load_json_file()
    extractor.process_data()
    extractor.save_to_json()

    # Step 3: AI Refinement (with logging)
    pipeline = AIPipeline(
        input_file="meta_data.json",
        output_file="results.json",
        log_file="pipeline.log",  # NEW: Log file
    )
    results = pipeline.process_file()

    # Step 4: Asset Management
    asset_manager = AssetManagement(
        input_file="results.json", output_file="assets.json"
    )
    asset_manager.load_results()
    asset_manager.process_assets()
    final_output_file = asset_manager.save_to_json()

    # Load final results (with asset_id)
    with open(final_output_file, "r") as f:
        final_results = json.load(f)

    # Step 5: Data Persistence (Simulated Azure Blob)
    persistence = DataPersistence()
    persisted_file = persistence.save_to_blob(final_results)

    return {
        "message": "Full pipeline executed: Preprocessing → Metadata → AI Refinement → Asset Management",
        "records_processed": len(final_results),
        "final_output_file": final_output_file,
        "persisted_file": persisted_file,
        "log_file": "pipeline.log",  # NEW: Return log file path
        "results": final_results,  # return all processed records
    }


# Retrieve Endpoint : fetch processed data from the pipeline
@app.api_route("/retrieve", methods=["GET", "POST"])
def retrieve_data(api_key: str = Depends(get_api_key), request: Request = None):
    """Return all processed records from assets.json, enforcing API key and rate limiting."""
    rate_limiter(request)

    try:
        with open("assets.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="No processed data found. Run /optimize first."
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error reading processed data.")

    return {
        "message": "Data retrieved successfully",
        "records": len(data),
        "data": data,
    }
