# import Libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from datetime import datetime
import os

# Helper function to map POS tags for lemmatization
def get_wordnet_pos(tag):
    """
    Return the WordNet POS tag for a given POS tag.
    """
    try:
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    except Exception as e:
        print(f"Error in get_wordnet_pos: {e}")
        return wordnet.NOUN


class DataProcessing:
    def __init__(self, data):
        """
        data : Convert to dataframe
        data: set a required columns
        """
        try:
            self.df = pd.DataFrame(data)
            self.columns_required = ["text", "rating", "timestamp"]
        except Exception as e:
            print(f"Error in __init__: {e}")

    def missing_values(self):
        """Handle missing values in rating, timestamp, and text"""
        try:
            # Fill missing ratings with mean
            self.df["rating"] = self.df["rating"].fillna(self.df["rating"].mean())

            # Fill missing timestamps with mode
            if self.df["timestamp"].isnull().any():
                mode_value = self.df["timestamp"].mode().iloc[0]
                self.df["timestamp"] = self.df["timestamp"].fillna(mode_value)

            # Fill missing text with placeholder
            self.df["text"] = self.df["text"].fillna("No Reviews")
            return self.df
        except Exception as e:
            print(f"Error in missing_values: {e}")
            return self.df

    def remove_stopwords_and_lemmatize(self):
        """Remove stopwords and apply lemmatization to 'text' column"""
        try:
            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()

            def clean_text(text):
                try:
                    if not isinstance(text, str):
                        return text
                    # Tokenize
                    words = word_tokenize(text)
                    # Remove stopwords
                    words = [w for w in words if w.lower() not in stop_words]
                    # POS tagging for accurate lemmatization
                    pos_tags = pos_tag(words)
                    # Lemmatize
                    lemmatized_words = [
                        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                        for word, tag in pos_tags
                    ]
                    return " ".join(lemmatized_words)
                except Exception as e:
                    print(f"Error in clean_text: {e}")
                    return text

            self.df["text"] = self.df["text"].apply(clean_text)
            return self.df
        except Exception as e:
            print(f"Error in remove_stopwords_and_lemmatize: {e}")
            return self.df

    def save_to_json(self, base_dir="output/cleaned"):
        """Save cleaned DataFrame to JSON file inside output/cleaned with timestamp"""
        try:
            os.makedirs(base_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(base_dir, f"cleaned_{timestamp}.json")

            self.df.to_json(filename, orient="records", lines=True)
            print(f"Data saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error in save_to_json: {e}")
            return None
