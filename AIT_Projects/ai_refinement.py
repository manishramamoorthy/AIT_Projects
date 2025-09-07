import os
import json
import re
import logging
from datetime import datetime

try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel
except Exception:
    torch = None
    DistilBertTokenizer = None
    DistilBertModel = None


class AIPipeline:
    def __init__(
        self,
        input_file: str = None,
        input_dir: str = "output/metadata_extraction",
        output_dir: str = "output/ai_refinement",
        log_file: str = "ai_pipeline.log",
    ):
        """
        AI refinement pipeline.
         - If input_file is provided, uses that file.
         - Otherwise, will look up latest .json in input_dir.
         - Writes output to output_dir/ai_refinement_{date}.json
        """
        self.input_file = input_file
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_file = log_file

        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.tokenizer = None
        self.model = None
        self.Q_table = {"good": 1, "bad": 0}

        # Try to load tokenizer & model, but fail gracefully
        try:
            if DistilBertTokenizer and DistilBertModel:
                self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            else:
                logging.warning("Transformers or torch not available; embeddings will be disabled.")
        except Exception as e:
            logging.error(f"[AIPipeline __init__] error loading model/tokenizer: {e}")
            self.tokenizer = None
            self.model = None

    def get_latest_input_file(self):
        """If input_file is None, find the latest .json file in input_dir."""
        try:
            if self.input_file:
                if os.path.exists(self.input_file):
                    return self.input_file
                else:
                    logging.error(f"[get_latest_input_file] Provided input_file not found: {self.input_file}")
                    return None

            files = [
                os.path.join(self.input_dir, f)
                for f in os.listdir(self.input_dir)
                if f.endswith(".json")
            ]
            if not files:
                logging.error(f"[get_latest_input_file] No JSON files found in {self.input_dir}")
                return None
            latest = max(files, key=os.path.getctime)
            self.input_file = latest
            return latest
        except Exception as e:
            logging.error(f"[get_latest_input_file] {e}")
            return None

    def generate_output(self, text: str):
        """Generate an embedding vector for the text. Returns list-of-list (or empty list on failure)."""
        try:
            if not self.tokenizer or not self.model:
                logging.warning("[generate_output] tokenizer/model not available; returning empty embedding")
                return []

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # shape (1, hidden_size)
            return embedding.numpy().tolist()
        except Exception as e:
            logging.error(f"[generate_output] {e}")
            return []

    def refine_output(self, embedding, reward_score):
        """Threshold-based refinement: reward >= 7 → good, else bad."""
        try:
            # ensure reward_score is numeric
            try:
                reward = float(reward_score)
            except Exception:
                reward = 0.0

            action = "good" if reward >= 7 else "bad"
            self.Q_table[action] += 0.1 * (reward - self.Q_table[action])

            return {
                "action": action,
                "reward": reward,
                "Q_values": self.Q_table.copy(),
            }
        except Exception as e:
            logging.error(f"[refine_output] {e}")
            return {"action": "error", "reward": reward_score, "Q_values": self.Q_table.copy()}

    def anonymize_text(self, text: str):
        """Simple anonymization: replace capitalized words with ANON (wrapped in try/except)."""
        try:
            return re.sub(r"\b[A-Z][a-z]*\b", "ANON", text)
        except Exception as e:
            logging.error(f"[anonymize_text] {e}")
            return text

    def build_output_filename(self, infile: str) -> str:
        """Build output filename with latest date from input file timestamp or current date."""
        try:
            # try from input file timestamp
            ts = os.path.getctime(infile)
            file_date = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
        except Exception:
            # fallback to now
            file_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        filename = f"ai_refinement_{file_date}.json"
        return os.path.join(self.output_dir, filename)

    def process_file(self):
        """Read input_file (JSON array), generate embeddings & refinements, save to output_file."""
        try:
            infile = self.get_latest_input_file()
            if not infile:
                logging.error("[process_file] No input file available")
                return None

            with open(infile, "r") as f:
                data = json.load(f)  # expecting list of dicts

            results = []
            for idx, record in enumerate(data, start=1):
                try:
                    text = record.get("text", "")
                    rating = record.get("rating", 5)

                    embedding = self.generate_output(text)
                    refined = self.refine_output(embedding, rating)
                    anonymized = self.anonymize_text(text)

                    result_entry = {
                        "id": idx,
                        "input_id": record.get("id"),
                        "original_text_anonymized": anonymized,
                        "embedding_size": len(embedding[0]) if embedding else 0,
                        "refined_output": refined,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                    results.append(result_entry)

                    logging.info(f"[process_file] processed idx={idx}, action={refined.get('action')}")
                except Exception as inner:
                    logging.error(f"[process_file] error processing record idx={idx}: {inner}")

            # ensure output dir exists
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                logging.error(f"[process_file] error creating output dir: {e}")

            # build output filename
            output_file = self.build_output_filename(infile)

            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            logging.info(f"[process_file] saved ai refinement results to {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"[process_file] {e}")
            return None

    def run(self):
        """Convenience wrapper to run the pipeline end-to-end."""
        try:
            return self.process_file()
        except Exception as e:
            logging.error(f"[run] {e}")
            return None
