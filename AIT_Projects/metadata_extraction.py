import nltk
import json
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# # Download NLTK data (run once)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


class MetaDataExtraction:
    def __init__(self, input_file="cleaned_data.json", output_file="final_output.json"):
        """
        Initialize with input and output JSON file paths
        """
        self.input_file = input_file
        self.output_file = output_file
        self.data = []
        self.output = []

    def load_json_file(self):
        """Load preprocessed JSON lines file"""
        with open(self.input_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def extract_names(self, text):
        """Extract PERSON entities from text using NLTK"""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)

        names = []
        for chunk in chunks:
            if isinstance(chunk, Tree) and chunk.label() == "PERSON":
                name = " ".join(c[0] for c in chunk.leaves())
                names.append(name)

        # If no name found, return "No Name"
        if not names:
            return ["No Name"]
        return names

    def process_data(self):
        """Process input data to extract metadata"""
        self.output = []
        for idx, record in enumerate(self.data, start=1):
            names = self.extract_names(record.get("text", ""))
            self.output.append(
                {
                    "id": idx,
                    "text": record.get("text", ""),
                    "names": names,
                    "rating": record.get("rating"),
                }
            )

    def save_to_json(self):
        """Save extracted metadata into JSON file"""
        with open(self.output_file, "w") as f:
            json.dump(self.output, f, indent=4)
        print(f"Final metadata saved to {self.output_file}")

    def run(self):
        """Convenience method for full pipeline"""
        self.load_json_file()
        self.process_data()
        self.save_to_json()
        return self.output
