"""Supervised sign-importance classifier (the T_i signal).

Task: given an OCR string, predict whether it is navigation-relevant signage
(EXIT, STOP, room numbers, restrooms, hazards) or decorative / advertising text.

Model: character n-gram TF-IDF + logistic regression. Character n-grams make the
model robust to the OCR corruptions that appear in blind-user photographs
(0/O, 1/I, missing letters), which a word-level model would miss.

Training data is generated from templates with OCR-style noise so the classifier
can be trained without any manual labelling; `data/text_importance/extra.csv`
(text,label) is merged in when present so hand-labelled real OCR output can be added.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple

from ..cues.text_importance import normalise_text

IMPORTANT = [
    "EXIT", "EMERGENCY EXIT", "FIRE EXIT", "EXIT ONLY", "STOP", "CAUTION", "CAUTION WET FLOOR", "WET FLOOR",
    "DANGER", "WARNING", "DO NOT ENTER", "NO ENTRY", "KEEP OUT", "STAIRS", "STAIRWAY", "ELEVATOR", "LIFT",
    "RESTROOM", "RESTROOMS", "TOILET", "MEN", "WOMEN", "ALL GENDER RESTROOM", "PUSH", "PULL", "ENTRANCE",
    "MAIN ENTRANCE", "OPEN", "CLOSED", "OUT OF ORDER", "ONE WAY", "BUS STOP", "PLATFORM 2", "GATE 12", "GATE B4",
    "ROOM 204", "ROOM 117B", "ROOM 305", "RM 101", "FLOOR 3", "LEVEL 2", "3RD FLOOR", "CROSSING", "WALK",
    "DONT WALK", "MIND THE GAP", "WATCH YOUR STEP", "AUTOMATIC DOOR", "EMERGENCY", "FIRST AID", "FIRE ALARM",
    "NO EXIT", "STAFF ONLY", "AUTHORIZED PERSONNEL ONLY", "HIGH VOLTAGE", "SLIPPERY WHEN WET", "LOW CLEARANCE",
    "STEP DOWN", "STEP UP", "RAMP", "HANDICAP ACCESSIBLE", "ACCESSIBLE ENTRANCE", "OFFICE 210", "LAB 12",
    "LECTURE HALL A", "CHECK IN", "INFORMATION", "RECEPTION", "TRACK 4", "TERMINAL B", "TAXI", "PHARMACY",
    "EMERGENCY ROOM", "ROOM 1204", "SUITE 300", "APT 4B", "A12", "B7", "204", "117B", "1102", "SECOND FLOOR",
]
DECOR = [
    "SALE", "50% OFF", "BIG SALE TODAY", "WELCOME", "WELCOME BACK STUDENTS", "MENU", "COFFEE $2", "LATTE $4.50",
    "FOLLOW US ON INSTAGRAM", "www.example.com", "SPRING CONCERT SATURDAY", "HAPPY HOUR 5 TO 7", "JOIN THE CLUB",
    "NEW MENU ITEMS", "BUY ONE GET ONE FREE", "ART SHOW FRIDAY", "FRESH BAKED DAILY", "GRAND OPENING",
    "LIVE MUSIC TONIGHT", "STUDENT DISCOUNT", "HIRING NOW", "BEST PIZZA IN TOWN", "LOVE WHERE YOU LIVE",
    "MOVIE NIGHT", "BAKE SALE", "TICKETS ON SALE", "VOTE FOR SAM", "LOST CAT", "YOGA CLASS", "TUTORING AVAILABLE",
    "SUMMER READING", "THE GREAT GATSBY", "CHAPTER ONE", "ONCE UPON A TIME", "PHOTOGRAPHY EXHIBIT",
    "CLUB FAIR WEDNESDAY", "GO BULLDOGS", "HAVE A NICE DAY", "SMILE", "DREAM BIG", "EST 1701", "LIMITED EDITION",
    "20 PERCENT OFF", "HALF PRICE DRINKS", "OPEN MIC NIGHT", "BOOK CLUB MEETS TUESDAY", "COMPUTER SCIENCE",
    "ANNUAL REPORT", "NEWSLETTER", "THANK YOU", "SPONSORED BY", "PRESENTED BY", "COMING SOON", "SEASON FINALE",
    "SUGAR FREE", "GLUTEN FREE", "ORGANIC", "PREMIUM QUALITY", "MADE IN USA", "NUTRITION FACTS", "INGREDIENTS",
    "LOREM IPSUM DOLOR", "PROFESSOR OFFICE HOURS", "PLEASE RECYCLE", "WIFI PASSWORD", "GUEST NETWORK",
]
CORRUPT = {"O": "0", "I": "1", "E": "F", "S": "5", "T": "7", "B": "8", "A": "4", "L": "I", "G": "6"}


def _noisy(s: str, rng: random.Random) -> str:
    chars = list(s)
    n = rng.choice([0, 0, 1, 1, 2])
    for _ in range(n):
        i = rng.randrange(len(chars))
        op = rng.random()
        if op < 0.5 and chars[i].upper() in CORRUPT:
            chars[i] = CORRUPT[chars[i].upper()]
        elif op < 0.75:
            chars[i] = ""            # dropped character
        else:
            chars[i] = chars[i] + " "  # OCR split
    out = "".join(chars)
    if rng.random() < 0.3:
        out = out.lower()
    return out


def generate_dataset(n_per_class: int = 600, seed: int = 0, extra_csv: str = "data/text_importance/extra.csv") -> Tuple[List[str], List[int]]:
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_per_class):
        X.append(normalise_text(_noisy(rng.choice(IMPORTANT), rng))); y.append(1)
        X.append(normalise_text(_noisy(rng.choice(DECOR), rng))); y.append(0)
    p = Path(extra_csv)
    if p.exists():
        with p.open() as f:
            for row in csv.DictReader(f):
                X.append(normalise_text(row["text"])); y.append(int(row["label"]))
    return X, y


def build_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")),
    ])


def train_and_save(out_path: str = "models/text_importance.joblib", seed: int = 0) -> dict:
    import joblib
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    X, y = generate_dataset(seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = build_model()
    cv = cross_val_score(build_model(), Xtr, ytr, cv=5, scoring="f1")
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    report = {
        "n_train": len(Xtr), "n_test": len(Xte),
        "cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std()),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_f1": float(f1_score(yte, pred)),
        "test_auc": float(roc_auc_score(yte, proba)),
    }
    # held-out phrases never seen in the templates
    unseen = [("EXIT 3", 1), ("R00M 218", 1), ("CAUT1ON", 1), ("STAIRS DOWN", 1), ("WOMENS RESTROOM", 1),
              ("FREE SAMPLES", 0), ("CONCERT TICKETS", 0), ("FOLLOW US", 0), ("BEST COFFEE", 0), ("JOIN US FRIDAY", 0)]
    ux = [normalise_text(s) for s, _ in unseen]
    up = model.predict_proba(ux)[:, 1]
    report["unseen_accuracy"] = float(sum(int(p >= 0.5) == l for p, (_, l) in zip(up, unseen)) / len(unseen))
    report["unseen_examples"] = [(s, round(float(p), 3)) for (s, _), p in zip(unseen, up)]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return report
