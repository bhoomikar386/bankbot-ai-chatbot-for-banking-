# nlu_engine/entity_extractor.py
import re, json, os
from typing import List, Dict
import spacy

BASE = os.path.dirname(__file__)
ENT_PATH = os.path.join(BASE, "entities.json")

class EntityExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        # try load spacy, else set to None
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

        if os.path.exists(ENT_PATH):
            with open(ENT_PATH, "r", encoding="utf-8") as f:
                self.patterns = json.load(f)
        else:
            self.patterns = {
                "amount": r"(?:\b|[\s,]|^)(?:₹|Rs\.?|INR|\$)\s?\d{1,3}(?:[,\d]{0,})?(?:\.\d{1,2})?\b",
                "account_number": r"\b\d{6,16}\b",
                "transaction_id": r"\b(?:TXN|TRN|UTR|REF)[-_]?[A-Za-z0-9]{3,}\b",
                "date": r"\b(?:today|tomorrow|yesterday|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
                "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
            }
        self.compiled = {k: re.compile(v, flags=re.IGNORECASE) for k,v in self.patterns.items()}

    def _reserve_span(self, reserved: List[tuple], start: int, end: int) -> bool:
        for s,e in reserved:
            if not (end <= s or start >= e):
                return False
        reserved.append((start, end))
        return True

    def _normalize_amount(self, raw: str):
        s = raw.replace("₹", "").replace("Rs", "").replace("INR", "").replace("$", "")
        s = s.replace(",", "").strip()
        try:
            if "." in s:
                return float(s)
            return float(int(s))
        except:
            return raw.strip()

    def extract_regex(self, text: str) -> List[Dict]:
        results = []
        reserved = []
        order = ["transaction_id", "account_number", "amount", "date", "ifsc"]
        for name in order:
            pat = self.compiled.get(name)
            if not pat:
                continue
            for m in pat.finditer(text):
                start, end = m.start(), m.end()
                if self._reserve_span(reserved, start, end):
                    val = m.group(0).strip()
                    item = {"entity": name, "value": val, "start": start, "end": end, "source": "regex"}
                    if name == "amount":
                        item["normalized"] = self._normalize_amount(val)
                    results.append(item)
        return results

    def extract_spacy(self, text: str) -> List[Dict]:
        if not self.nlp:
            return []
        doc = self.nlp(text)
        out = []
        reserved = []
        for ent in doc.ents:
            start, end = ent.start_char, ent.end_char
            overlaps = any(not (end <= s or start >= e) for s,e in reserved)
            if overlaps:
                continue
            out.append({"entity": ent.label_.lower(), "value": ent.text, "start": start, "end": end, "source": "spacy"})
            reserved.append((start, end))
        return out

    def extract(self, text: str) -> List[Dict]:
        text = text or ""
        regex_ents = self.extract_regex(text)
        spacy_ents = self.extract_spacy(text)
        reserved = [(e["start"], e["end"]) for e in regex_ents]
        for e in spacy_ents:
            if any(not (e['end'] <= s or e['start'] >= t) for s,t in reserved):
                continue
            regex_ents.append(e)
            reserved.append((e['start'], e['end']))
        regex_ents.sort(key=lambda x: x["start"])
        return regex_ents

if __name__ == "__main__":
    ex = EntityExtractor()
    s = "Please transfer ₹2,500 to account 9988776655. TXN: TXN12345. Do it tomorrow."
    print(ex.extract(s))
