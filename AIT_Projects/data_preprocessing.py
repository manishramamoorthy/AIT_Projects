# import Libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


# Helper function to map POS tags for lemmatization
def get_wordnet_pos(tag):
    """
    Return the WordNet POS tag for a given POS tag.
    """
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


class DataProcessing:
    def __init__(self, data):
        """
        data : Convert to dataframe
        data: set a required columns
        """
        self.df = pd.DataFrame(data)
        self.columns_required = ["text", "rating", "timestamp"]


    def missing_values(self):
        """Handle missing values in rating, timestamp, and text"""
        # Fill missing ratings with mean
        self.df["rating"] = self.df["rating"].fillna(self.df["rating"].mean())

        # Fill missing timestamps with mode
        if self.df["timestamp"].isnull().any():
            mode_value = self.df["timestamp"].mode().iloc[0]
            self.df["timestamp"] = self.df["timestamp"].fillna(mode_value)

        # Fill missing text with placeholder
        self.df["text"] = self.df["text"].fillna("No Reviews")
        return self.df

    def remove_stopwords_and_lemmatize(self):
        """Remove stopwords and apply lemmatization to 'text' column"""
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()


        def clean_text(text):
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

        self.df["text"] = self.df["text"].apply(clean_text)
        return self.df

    def save_to_json(self, filename="cleaned_data.json"):
        """Save cleaned DataFrame to JSON file"""
        self.df.to_json(filename, orient="records", lines=True)
        print(f"Data saved to {filename}")
