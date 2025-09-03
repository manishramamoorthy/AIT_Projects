from transformers import DistilBertTokenizer, DistilBertModel
import torch
import re
import random
import json
import logging
from datetime import datetime


class AIPipeline:
    def __init__(
        self,
        input_file="meta_data.json",
        output_file="results.json",
        log_file="pipeline.log",
    ):
        """
        Initialize with input, output JSON file  and log file paths.
        Initialize a tokenize data, hugging face model and mapping content
        """
        self.input_file = input_file
        self.output_file = output_file
        self.log_file = log_file

        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.Q_table = {"good": 0, "bad": 0}


    def generate_output(self, text):
        """Generate and return an embedding vector for the given text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.numpy().tolist()


    def refine_output(self, embedding, reward_score):
        """Update Q-values using reward feedback and return the chosen action."""
        action = random.choice(["good", "bad"])
        self.Q_table[action] += 0.1 * (reward_score - self.Q_table[action])
        return {
            "action": action,
            "reward": reward_score,
            "Q_values": self.Q_table.copy(),
        }


    def anonymize_text(self, text):
        """Regex used to Replace capitalized words in the text with 'ANON'."""
        return re.sub(r"\b[A-Z][a-z]*\b", "ANON", text)


    def process_file(self):
        """Process input JSON, generate embeddings, refine outputs, anonymize text, and save results."""
        with open(self.input_file, "r") as f:
            data = json.load(f)

        results = []
        for idx, record in enumerate(data, start=1):
            text = record.get("text", "")
            reward = record.get("rating", 5)

            output = self.generate_output(text)
            refined = self.refine_output(output, reward)
            anonymized_text = self.anonymize_text(text)

            result_entry = {
                "id": idx,
                "original_text_anonymized": anonymized_text,
                "embedding_size": len(output[0]),
                "refined_output": refined,
            }
            results.append(result_entry)

            # Write log entry (with anonymized text only for privacy)
            logging.info(
                f"Processed ID={idx}, Action={refined['action']}, "
                f"Reward={refined['reward']}, AnonymizedText={anonymized_text[:50]}..."
            )

        # Save final results in JSON
        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=4)

        return results
