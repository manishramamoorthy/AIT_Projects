import os
import json
import datetime
from glob import glob
import logging


class DataPersistence:
    def __init__(self, input_dir="output/asset_management", container="mock_blob_storage"):
        """Initialize storage with a local folder as the container and set input dir."""
        try:
            self.input_dir = input_dir
            self.container = container
            os.makedirs(container, exist_ok=True)
        except Exception as e:
            logging.error(f"[__init__] Error initializing DataPersistence: {e}")
            raise

    def get_latest_input_file(self):
        """Fetch the latest asset_management JSON file."""
        try:
            files = glob(os.path.join(self.input_dir, "asset_management_*.json"))
            if not files:
                raise FileNotFoundError(f"No asset_management files found in {self.input_dir}")
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
        except Exception as e:
            logging.error(f"[get_latest_input_file] {e}")
            raise

    def load_input_data(self, file_path):
        """Load JSON data from the given file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"[load_input_data] Failed to load file {file_path}: {e}")
            raise

    def save_to_blob(self, data, blob_name=None):
        """Simulate saving data to Azure Blob Storage (local folder mock)."""
        try:
            if not blob_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                blob_name = f"final_data_{timestamp}.blob.json"

            blob_path = os.path.join(self.container, blob_name)
            with open(blob_path, "w") as f:
                json.dump(data, f, indent=4)

            logging.info(f"Data persisted to simulated Azure Blob: {blob_path}")
            return blob_path
        except Exception as e:
            logging.error(f"[save_to_blob] Error persisting data: {e}")
            return None

    def run(self):
        """Main pipeline: get latest input -> load -> save to blob."""
        try:
            latest_file = self.get_latest_input_file()
            data = self.load_input_data(latest_file)
            return self.save_to_blob(data)
        except Exception as e:
            logging.error(f"[run] Failed: {e}")
            return None
