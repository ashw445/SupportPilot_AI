import os
import re
import joblib

MODEL_PATH = os.path.join("models", "ticket_classifier.pkl")
CONFIDENCE_THRESHOLD = 0.2

SEVERITY_RULES = {
    "Critical": ["server down", "entire company", "production down", "data breach", "outage", "system unavailable"],
    "High": ["urgent", "cannot work", "business stopped", "client meeting", "vpn", "completely unusable"],
    "Medium": ["slow", "error", "problem", "issue", "failed", "stuck", "crashes", "crashing"],
}

PRIORITY_MAP = {
    "Critical": "P1",
    "High": "P2",
    "Medium": "P3",
    "Low": "P4",
}


def clean_text(text: str) -> str:
    """Shared text normalization for both training and inference."""
    cleaned = re.sub(r"[^a-z0-9_\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def predict_severity(text: str) -> str:
    normalized_text = text.lower()
    for severity, keywords in SEVERITY_RULES.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized_text):
                return severity
    return "Low"


def classify_text(description: str, model, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict | None:
    if model is None:
        raise ValueError("Classification model is not loaded.")

    cleaned_text = clean_text(description)
    if not cleaned_text:
        return None

    probs = model.predict_proba([cleaned_text])[0]
    best_idx = probs.argmax()
    confidence = round(float(probs[best_idx]), 2)

    category = "Uncategorized" if confidence < confidence_threshold else str(model.classes_[best_idx])
    severity = predict_severity(cleaned_text)
    priority = PRIORITY_MAP.get(severity, "P4")

    return {
        "category": category,
        "confidence": confidence,
        "priority": priority,
        "severity": severity,
    }


if __name__ == "__main__":
    try:
        classification_model = joblib.load(MODEL_PATH)
        sample = "VPN is down and I cannot work before my client meeting"
        print("Model loaded successfully. Sample output:", classify_text(sample, classification_model))
    except FileNotFoundError:
        print(f"File {MODEL_PATH} not found. Run 'python train_model.py' first.")