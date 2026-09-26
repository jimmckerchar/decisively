"""Export a Laya checkpoint to ONNX and check ONNX Runtime matches PyTorch.

Usage: python script/laya/export.py <english|multilingual|typed-decisions> <out_dir>
Writes <out_dir>/model.onnx (plus model.onnx.data for checkpoints over ONNX's 2 GB limit) and the
tokenizer and rl_agent_config.json Layar::Laya needs. Then run script/laya/parity.py.
"""
import os, shutil, sys, time, warnings
warnings.filterwarnings("ignore")

import torch
import laya
from laya.common import collate_items

name, out_dir = sys.argv[1], sys.argv[2]
sub = None if name == "english" else name
os.makedirs(out_dir, exist_ok=True)

agent = laya.load("convaiinnovations/laya", subfolder=sub, device="cpu")
model = agent.model.eval()
# The fused TransformerEncoderLayer fast path (aten::_transformer_encoder_layer_fwd) has no ONNX op.
torch.backends.mha.set_fastpath_enabled(False)
try:
    model.encoder.config._attn_implementation = "eager"   # sdpa/flash paths don't export cleanly
except Exception:
    pass


def batch_for(state, questions):
    ids = list(questions)
    internal = {q: agent._to_internal(questions[q]) for q in ids}
    items = agent._encode_state(state, ids, internal)
    b = collate_items([items], agent.tok.pad_token_id)
    return {k: b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


example = batch_for("I was charged twice for my subscription this month", {
    "a": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "bug", "account"]},
    "b": {"type": "noul", "instructions": "Is the customer angry?"},
})

onnx_path = os.path.join(out_dir, "model.onnx")
tokens = torch.export.Dim("tokens", min=8, max=8192)
options = torch.export.Dim("options", min=1, max=255)
questions = torch.export.Dim("questions", min=1, max=64)
t = time.time()
with torch.no_grad():
    torch.onnx.export(
        model,
        tuple(example[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")),
        onnx_path,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        # torch.export with symbolic dims: the TorchScript exporter baked the example's
        # sequence length into the head's attention reshapes.
        dynamic_shapes={
            "input_ids": {0: questions, 1: tokens},
            "attention_mask": {0: questions, 1: tokens},
            "marker_pos": {0: questions, 1: options},
            "marker_mask": {0: questions, 1: options},
            "qtype": {0: questions},
        },
        opset_version=18,
        dynamo=True,
    )
size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if f.startswith("model.onnx"))
print(f"exported {name} in {time.time() - t:.0f}s -> {size / 1e6:.0f} MB")

# Ship what the Ruby side needs next to the model.
src = os.path.join(agent.tok.name_or_path)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), out_dir)
shutil.copy(os.path.join(os.path.dirname(src), "rl_agent_config.json"), out_dir)

print(f"next: python script/laya/parity.py {name} {out_dir}")
