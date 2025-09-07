
import os
import time
import logging
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from data_preprocessing import DataProcessing
from metadata_extraction import MetaDataExtraction
from ai_refinement import AIPipeline
from asset_management import AssetManagement
from data_persistence import DataPersistence   # ✅ NEW

load_dotenv()

# Configure app + logging
app = FastAPI(title="Data Optimization & Management Microservice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# API key config
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = os.getenv("API_KEY_NAME", "x-api-key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Rate limiting config
RATE_LIMIT = 5
TIME_WINDOW = 60
request_log: Dict[str, list] = {}


def get_api_key(api_key: str = Depends(api_key_header)):
    try:
        if api_key != API_KEY:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")
        return api_key
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")


def rate_limiter(request: Request):
    client_ip = request.client.host
    now = time.time()

    if client_ip not in request_log:
        request_log[client_ip] = []
    request_log[client_ip] = [t for t in request_log[client_ip] if now - t < TIME_WINDOW]

    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too Many Requests. Try again later.")

    request_log[client_ip].append(now)


class Review(BaseModel):
    rating: Optional[float] = None
    timestamp: Optional[str] = None
    text: Optional[str] = None


@app.post("/optimize")
def process_pipeline(
    reviews: List[Review],
    api_key: str = Depends(get_api_key),
    _: None = Depends(rate_limiter),
):
    """
    Runs the pipeline:
    1. DataPreprocessing
    2. MetadataExtraction
    3. AI Refinement
    4. AssetManagement
    5. DataPersistence
    """
    if not reviews:
        raise HTTPException(status_code=400, detail="No data provided")

    try:
        # STEP 1: Data Preprocessing
        try:
            data_list = [r.dict() for r in reviews]
            processor = DataProcessing(data_list)
            processor.missing_values()
            processor.remove_stopwords_and_lemmatize()
            cleaned_file = processor.save_to_json()
            if not cleaned_file:
                raise RuntimeError("DataProcessing.save_to_json returned None/empty")
        except Exception as e:
            logging.exception("Data preprocessing failed")
            raise HTTPException(status_code=500, detail=f"Data preprocessing failed: {e}")

        # STEP 2: Metadata Extraction
        try:
            extractor = MetaDataExtraction(input_dir="output/cleaned", output_dir="output/metadata_extraction")
            metadata_output = extractor.run()
            metadata_file = extractor.output_file
            if not metadata_output or not metadata_file:
                raise RuntimeError("Metadata extraction produced no output")
        except Exception as e:
            logging.exception("Metadata extraction failed")
            raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {e}")

        # STEP 3: AI Refinement
        try:
            ai_pipeline = AIPipeline(input_file=metadata_file, output_dir="output/ai_refinement")
            ai_output_file = ai_pipeline.run()
            if not ai_output_file:
                raise RuntimeError("AI refinement failed or produced no output")
        except Exception as e:
            logging.exception("AI refinement failed")
            raise HTTPException(status_code=500, detail=f"AI refinement failed: {e}")

        # STEP 4: Asset Management
        try:
            asset_manager = AssetManagement(input_dir="output/ai_refinement", output_dir="output/asset_management")
            asset_output_file = asset_manager.run()
            if not asset_output_file:
                raise RuntimeError("Asset management failed or produced no output")
        except Exception as e:
            logging.exception("Asset management failed")
            raise HTTPException(status_code=500, detail=f"Asset management failed: {e}")

        # STEP 5: Data Persistence
        try:
            persistence = DataPersistence(input_dir="output/asset_management", container="mock_blob_storage")
            blob_file = persistence.run()
            if not blob_file:
                raise RuntimeError("Data persistence failed or produced no output")
        except Exception as e:
            logging.exception("Data persistence failed")
            raise HTTPException(status_code=500, detail=f"Data persistence failed: {e}")

        # Final response
        return {
            "message": "Pipeline completed successfully",
            "cleaned_file": cleaned_file,
            "metadata_file": metadata_file,
            "ai_refinement_file": ai_output_file,
            "asset_management_file": asset_output_file,
            "persisted_file": blob_file,
            "records": len(data_list),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Unexpected pipeline error")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
