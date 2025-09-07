import json
import os
from datetime import datetime


class AssetManagement:
    def __init__(self, input_dir="output/ai_refinement", output_dir="output/asset_management"):
        """Initialize with input/output directories instead of fixed files."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_file = None
        self.output_file = None
        self.data = []

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def get_latest_input_file(self):
        """Find the latest ai_refinement_{date}.json file."""
        files = [
            f for f in os.listdir(self.input_dir)
            if f.startswith("ai_refinement_") and f.endswith(".json")
        ]
        if not files:
            raise FileNotFoundError("No ai_refinement JSON files found.")

        latest_file = max(
            files,
            key=lambda x: os.path.getmtime(os.path.join(self.input_dir, x))
        )
        self.input_file = os.path.join(self.input_dir, latest_file)

        # Extract the timestamp from the filename
        timestamp = latest_file.replace("ai_refinement_", "").replace(".json", "")
        self.output_file = os.path.join(self.output_dir, f"asset_management_{timestamp}.json")

    def load_results(self):
        """Load input JSON file (AI refinement output)."""
        if not self.input_file or not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file {self.input_file} not found.")

        with open(self.input_file, "r") as f:
            self.data = json.load(f)

        # Ensure data is always a list
        if isinstance(self.data, dict):
            self.data = [self.data]

    def process_assets(self):
        """Assign unique asset IDs to each record."""
        if not self.data:
            raise ValueError("No data loaded. Run load_results() first.")

        for idx, record in enumerate(self.data, start=1):
            record["asset_id"] = f"asset_{idx:03d}"

    def save_to_json(self):
        """Save updated data with asset IDs to file."""
        if not self.data:
            raise ValueError("No processed data to save.")

        with open(self.output_file, "w") as f:
            json.dump(self.data, f, indent=4)

        print(f"Asset IDs assigned and saved to {self.output_file}")
        return self.output_file

    def run(self):
        """Run the full asset management pipeline with error handling."""
        try:
            self.get_latest_input_file()
            self.load_results()
            self.process_assets()
            return self.save_to_json()
        except Exception as e:
            print(f"Asset Management failed: {e}")
            return None
