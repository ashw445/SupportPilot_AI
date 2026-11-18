import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from classifier import clean_text

MODEL_PATH = os.path.join("models", "ticket_classifier.pkl")
CSV_PATH = "tickets.csv"


def train():
    if not os.path.exists(CSV_PATH):
        print(f"Dataset '{CSV_PATH}' not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    required_cols = {"Ticket", "Category"}
    if not required_cols.issubset(df.columns):
        print(f"CSV must contain columns: {required_cols}")
        sys.exit(1)

    df = df.dropna(subset=["Ticket", "Category"]).copy()
    df["Ticket"] = df["Ticket"].apply(clean_text)
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df[(df["Ticket"] != "") & (df["Category"] != "")]

    if len(df) < 2:
        print("Not enough valid rows to train a model.")
        sys.exit(1)

    # Only stratify if every category has at least 2 samples
    class_counts = df["Category"].value_counts()
    stratify_target = df["Category"] if class_counts.min() >= 2 else None

    X_train, X_test, Y_train, Y_test = train_test_split(
        df["Ticket"],
        df["Category"],
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    pipeline = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2 if len(X_train) >= 10 else 1,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)

    print("Model Training complete.")
    print(f"Accuracy Score: {accuracy_score(Y_test, Y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()