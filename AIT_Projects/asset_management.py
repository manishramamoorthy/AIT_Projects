import json
import os


class AssetManagement:
    def __init__(self, input_file="results.json", output_file="assets.json"):
        """Initialize with input and output JSON file paths"""
        self.input_file = input_file
        self.output_file = output_file
        self.data = []


    def load_results(self):
        """Load input JSON file (AI refinement output)."""
        if not os.path.exists(self.input_file):
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
