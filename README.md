# Data Optimization & Management Microservice

A FastAPI-based microservice for end-to-end processing of review data, including preprocessing, metadata extraction, AI refinement, asset management, and data persistence.

---

## 📝 Features

1. **Data Preprocessing**
   - Handles missing values in text, rating, and timestamp.
   - Removes stopwords and applies lemmatization.
   - Saves cleaned data to JSON.

2. **Metadata Extraction**
   - Extracts named entities (e.g., PERSON) from text using NLTK.
   - Saves extracted metadata to JSON.

3. **AI Refinement**
   - Generates embeddings using DistilBERT.
   - Applies reinforcement-learning style refinement (Q-values).
   - Anonymizes sensitive information.
   - Logs processed records in a `.log` file for auditing.

4. **Asset Management**
   - Assigns unique `asset_id`s to each record.
   - Saves asset-managed data to JSON.

5. **Data Persistence**
   - Simulates Azure Blob storage using a local folder.
   - Saves final processed data with timestamped blob names.

---

## ⚙️ Tech Stack

- **Framework:** FastAPI
- **ML/NLP:** Hugging Face Transformers (DistilBERT), NLTK
- **Persistence:** JSON files and simulated Azure Blob
- **Logging:** Python `logging` module
- **Python Version:** 3.12+
