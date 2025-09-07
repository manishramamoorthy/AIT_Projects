import os
import json
import nltk
from datetime import datetime
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Make sure NLTK resources are available
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


class MetaDataExtraction:
    def __init__(self, input_dir="output/cleaned", output_dir="output/metadata_extraction"):
        """
        Initialize with input and output directories.
        The latest file from input_dir will be used automatically.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_file = None
        self.output_file = None
        self.data = []
        self.output = []

    def get_latest_input_file(self):
        """Get the latest cleaned JSON file from input_dir"""
        try:
            files = [
                os.path.join(self.input_dir, f)
                for f in os.listdir(self.input_dir)
                if f.startswith("cleaned_") and f.endswith(".json")
            ]
            if not files:
                raise FileNotFoundError("No cleaned JSON file found in input directory.")

            self.input_file = max(files, key=os.path.getctime)

            # Prepare output path
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = os.path.join(
                self.output_dir, f"metadata_{timestamp}.json"
            )
            return self.input_file
        except Exception as e:
            print(f"[Error in get_latest_input_file] {e}")
            return None

    def load_json_file(self):
        """Load preprocessed JSON lines file"""
        try:
            with open(self.input_file, "r") as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        except Exception as e:
            print(f"[Error in load_json_file] {e}")

    def extract_entities(self, text):
        """Extract names, occupations, and locations dynamically"""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)

            names, locations, occupations = [], [], []

            # Named Entity Recognition
            for chunk in chunks:
                if isinstance(chunk, Tree):
                    label = chunk.label()
                    entity = " ".join(c[0] for c in chunk.leaves())

                    if label == "PERSON":
                        names.append(entity)
                    elif label == "GPE":  # Location
                        locations.append(entity)
                    elif label == "ORGANIZATION":  # Can be occupation/org
                        occupations.append(entity)

            # Pattern-based occupation detection ("from HR", "in Marketing")
            for i, (word, tag) in enumerate(pos_tags):
                if i > 0 and pos_tags[i - 1][0].lower() in {"from", "in"}:
                    if tag.startswith("NN"):  # Noun is likely occupation
                        occupations.append(word)

            # Fallback: keyword-based detection
            occupation_keywords = {"HR", "Marketing", "Engineering", "Doctor", "Teacher", "Manager", "Developer"}
            for word in tokens:
                if word in occupation_keywords:
                    occupations.append(word)

            # Ensure occupation is compulsory
            if not occupations:
                occupations = ["Unknown Occupation"]

            return {
                "names": names if names else ["No Name"],
                "occupations": occupations,
                "locations": locations if locations else ["No Location"],
            }

        except Exception as e:
            print(f"[Error in extract_entities] {e}")
            return {
                "names": ["Error"],
                "occupations": ["Unknown Occupation"],
                "locations": ["Error"]
            }

    def process_data(self):
        """Process input data to extract metadata"""
        try:
            self.output = []
            for idx, record in enumerate(self.data, start=1):
                entities = self.extract_entities(record.get("text", ""))
                self.output.append(
                    {
                        "id": idx,
                        "text": record.get("text", ""),
                        "names": entities["names"],
                        "occupations": entities["occupations"],
                        "locations": entities["locations"],
                        "rating": record.get("rating"),
                    }
                )
        except Exception as e:
            print(f"[Error in process_data] {e}")

    def save_to_json(self):
        """Save extracted metadata into JSON file"""
        try:
            with open(self.output_file, "w") as f:
                json.dump(self.output, f, indent=4)
            print(f"Final metadata saved to {self.output_file}")
        except Exception as e:
            print(f"[Error in save_to_json] {e}")

    def run(self):
        """Convenience method for full pipeline"""
        try:
            if not self.get_latest_input_file():
                return None
            self.load_json_file()
            self.process_data()
            self.save_to_json()
            return self.output
        except Exception as e:
            print(f"[Error in run] {e}")
            return None
