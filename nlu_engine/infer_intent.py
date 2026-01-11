# nlu_engine/infer_intent.py
import os, json
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "intent_model")

class IntentClassifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}. Train first.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        # load labels if present
        labels_path = os.path.join(self.model_dir, "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            # fallback to model config labels
            num_labels = self.model.config.num_labels
            self.labels = [f"label_{i}" for i in range(num_labels)]
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        out = self.pipe(text)
        if isinstance(out, list) and len(out) > 0:
            scores = out[0]
            sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
            preds = []
            for s in sorted_scores:
                lab = s["label"]
                # HF often returns 'LABEL_0' — map to our labels if possible
                if lab.startswith("LABEL_") and lab[6:].isdigit():
                    idx = int(lab[6:])
                    mapped = self.labels[idx] if idx < len(self.labels) else lab
                else:
                    mapped = lab
                preds.append({"intent": mapped, "confidence": float(s["score"])})
            return {"text": text, "predictions": preds}
        return {"text": text, "predictions": []}

if __name__ == "__main__":
    ic = IntentClassifier()
    print(ic.predict("Please transfer ₹2,500 to account 9988776655", top_k=5))