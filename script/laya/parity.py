"""Check ONNX Runtime output matches PyTorch for an exported Laya checkpoint.

Usage: python script/laya/parity.py <english|multilingual|typed-decisions> <out_dir>
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import laya
from laya.common import collate_items

name, out_dir = sys.argv[1], sys.argv[2]
sub = None if name == "english" else name
onnx_path = out_dir + "/model.onnx"
agent = laya.load("convaiinnovations/laya", subfolder=sub, device="cpu")


def batch_for(state, questions):
    ids = list(questions)
    internal = {q: agent._to_internal(questions[q]) for q in ids}
    items = agent._encode_state(state, ids, internal)
    b = collate_items([items], agent.tok.pad_token_id)
    return {k: b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
cases = [
    ("single choice, 8 options", "Hi, can you send me the invoice for last month?",
     {"q": {"type": "choice", "instructions": "What is this about?",
            "criteria": ["billing", "bug", "feature_request", "account", "shipping", "refunds", "security", "other"]}}),
    ("mixed batch", "URGENT: your account has been suspended. Verify your password at http://secure-login.co",
     {"spam": {"type": "noul", "instructions": "Is this message spam, phishing or a scam?"},
      "team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "payments", "security": "account safety"}},
      "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high", "critical"]}}),
    ("long state", "word " * 900, {"q": {"type": "noul", "instructions": "Is this repetitive?"}}),
]
worst = 0.0
for label, state, qs in cases:
    b = batch_for(state, qs)
    with torch.no_grad():
        pt_logits, pt_act = agent.model(*(b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    ox_logits, ox_act = sess.run(None, {
        "input_ids": b["input_ids"].numpy(), "attention_mask": b["attention_mask"].numpy(),
        "marker_pos": b["marker_pos"].numpy(), "marker_mask": b["marker_mask"].numpy(),
        "qtype": b["qtype"].numpy(),
    })
    # Compare what callers see: per-question option probabilities and act probabilities.
    # (Raw act logits run into the thousands, so an absolute logit tolerance is meaningless.)
    def softmax(z):
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    mask = b["marker_mask"].numpy()
    d = max(np.abs(softmax(pt_logits.numpy()[r, :mask[r].sum()]) - softmax(ox_logits[r, :mask[r].sum()])).max()
            for r in range(mask.shape[0]))
    d = max(d, np.abs(softmax(pt_act.numpy()) - softmax(ox_act)).max())
    worst = max(worst, d)
    print(f"  {label:26} tokens={b['input_ids'].shape[1]:4}  max prob diff = {d:.2e}")
print(f"PARITY {'OK' if worst < 1e-4 else 'FAILED'} (worst {worst:.2e})")
