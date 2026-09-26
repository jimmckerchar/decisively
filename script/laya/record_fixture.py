"""Record Python Laya's token ids and probabilities for cases the Ruby port must reproduce.

Usage: python script/laya/record_fixture.py <english|multilingual> spec/fixtures/laya/<checkpoint>.json
The Ruby side replays these in spec/laya_real_model_spec.rb.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import laya
from laya.common import QTYPES, collate_items, temp_bucket

name, out = sys.argv[1], sys.argv[2]
agent = laya.load("convaiinnovations/laya", subfolder=None if name == "english" else name, device="cpu")
mask = agent.tok.mask_token
long_words = " ".join(f"word{i}" for i in range(1500))

cases = [
    ("plain choice", "I was charged twice for my subscription this month",
     {"team": {"type": "choice", "instructions": "What is this message about?",
               "criteria": ["billing", "bug", "feature_request", "account"]}}),
    ("described choice", "Hi, can you send me the invoice for last month?",
     {"team": {"type": "choice", "instructions": "Which team should handle this message?",
               "criteria": {"billing": "payments, invoices, refunds", "bug": "something is broken",
                            "feature_request": "asking for new functionality", "account": None}}}),
    ("noul default", "CONGRATS!!! You won a free cruise, click here",
     {"spam": {"type": "noul", "instructions": "Is this message spam, phishing or a scam?"}}),
    ("noul described", "Please reset my password, I can't log in to my account.",
     {"spam": {"type": "noul", "instructions": "Is this a phishing attempt?",
               "criteria": {"true": "someone is trying to steal credentials", "false": "a genuine request"}}}),
    ("score", "This is the third time I've asked. Fix it now.",
     {"anger": {"type": "score", "instructions": "How angry is the customer?",
                "criteria": ["calm or happy", "mildly annoyed", "frustrated", "furious"]}}),
    ("mixed batch", "URGENT: your account has been suspended. Verify your password at http://secure-login.co",
     {"spam": {"type": "noul", "instructions": "Is this message spam, phishing or a scam?"},
      "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "security": "account safety"}},
      "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high", "critical"]}}),
    ("hash state, non-ascii", {"subject": "Doppelte Abbuchung", "body": "Ich wurde zweimal belastet — bitte erstatten 😊", "n": 2},
     {"team": {"type": "choice", "instructions": "Welches Team?", "criteria": ["billing", "bug", "account"]}}),
    ("json criteria", "The app crashes when I open settings",
     {"team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"bug": {"desc": "crashes, errors", "examples": ["crash", "500"]}, "billing": {"desc": "money"}}}}),
    ("long string state", "I was charged twice. " + long_words,
     {"q": {"type": "noul", "instructions": "Does the customer mention a double charge?"}}),
    ("conversation state", [f"user: message {i}" for i in range(400)] + ["user: I was charged twice, refund me"],
     {"q": {"type": "choice", "instructions": "What is the latest message about?", "criteria": ["billing", "bug", "other"]}}),
    ("many long options", "My parcel never arrived",
     {"q": {"type": "choice", "instructions": "Which category?",
            "criteria": {f"cat{i}": " ".join(["a detailed description of category", str(i)] * 8) for i in range(12)}}}),
    ("mask tokens in text", f"please {mask} refund {mask}",
     {"q": {"type": "choice", "instructions": f"Which {mask} team?", "criteria": [f"billing {mask}", "bug"]}}),
]

records = []
for label, state, questions in cases:
    ids = list(questions)
    internal = {q: agent._to_internal(questions[q]) for q in ids}
    items = agent._encode_state(state, ids, internal)
    b = collate_items([items], agent.tok.pad_token_id)
    with torch.no_grad():
        logits, _ = agent._forward(b)
    probs = {}
    for r, qid in enumerate(ids):
        k = len(items[r]["markers"])
        qt = QTYPES[internal[qid]["t"]]
        t = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        z = logits[r, :k] / t
        p = np.exp(z - z.max())
        probs[qid] = (p / p.sum()).tolist()
    records.append({
        "name": label, "state": state, "questions": questions,
        "input_ids": {qid: items[r]["ids"] for r, qid in enumerate(ids)},
        "markers": {qid: items[r]["markers"] for r, qid in enumerate(ids)},
        "probabilities": probs,
    })
    print(f"{label:22} " + "  ".join(f"{q}: {np.round(v, 3).tolist()}" for q, v in probs.items()))

with open(out, "w") as f:
    json.dump({"checkpoint": name, "cases": records}, f, ensure_ascii=False)
