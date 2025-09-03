import os
import json
import datetime


class DataPersistence:
    def __init__(self, container="mock_blob_storage"):
        """Initialize the storage with a local folder as the container."""
        self.container = container
        os.makedirs(container, exist_ok=True)

    def save_to_blob(self, data, blob_name=None):
        """Simulate saving data to Azure Blob Storage (local folder mock)."""
        if not blob_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            blob_name = f"final_data_{timestamp}.blob.json"

        blob_path = os.path.join(self.container, blob_name)
        try:
            with open(blob_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Data persisted to simulated Azure Blob: {blob_path}")
            return blob_path
        except Exception as e:
            print(f"Error persisting data: {e}")
            return None
