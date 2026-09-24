"""Fact-check helper: does Laya drop the tail of a long state? Live /decide on 1237.
Same question, two different judged snippets, placed AFTER n lines of preamble.
If the snippet is cut off, both snippets score identically."""
import json, urllib.request
from transformers import AutoTokenizer

SNAP = r"C:\Users\jwals\.cache\huggingface\hub\models--convaiinnovations--laya\snapshots\1c5edc17a7acd8701df6fc341c0d179f1c62c982\tokenizer"
tok = AutoTokenizer.from_pretrained(SNAP)

RUST = "fn add(a: i32, b: i32) -> i32 { a + b }"
TS = "function add(a: number, b: number): number { return a + b; }"
Q = {"lang": {"type": "choice", "instructions": "Which language is the code in the state written in?",
              "criteria": {"rust": "Rust source code", "typescript": "TypeScript source code"}}}
LINE = "log line {i}: request handled, status ok"


def decide(state):
    body = json.dumps({"state": state, "questions": Q}).encode()
    req = urllib.request.Request("http://127.0.0.1:1237/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        a = json.loads(r.read().decode())["answers"]["lang"]
    return a.get("probabilities") or a


print("tokens per preamble line:", len(tok(LINE.format(i=50), add_special_tokens=False)["input_ids"]) + 1)
for n in (0, 10, 20, 30, 40, 60, 80, 120):
    pre = "\n".join(LINE.format(i=i) for i in range(n))
    rows = []
    for name, code in (("rust", RUST), ("ts", TS)):
        state = (pre + "\n" + code) if n else code
        ntok = len(tok(state, add_special_tokens=False)["input_ids"])
        rows.append((name, ntok, decide(state)))
    same = rows[0][2] == rows[1][2]
    print(f"preamble {n:3d} lines  state_tokens {rows[0][1]:4d}/{rows[1][1]:4d}  "
          f"rust->{json.dumps(rows[0][2])}  ts->{json.dumps(rows[1][2])}  identical={same}")
# thing-first control at the longest preamble
pre = "\n".join(LINE.format(i=i) for i in range(120))
print("thing FIRST, 120 lines after:", "rust->", json.dumps(decide(RUST + "\n" + pre)),
      " ts->", json.dumps(decide(TS + "\n" + pre)))
