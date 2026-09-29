#!/usr/bin/env python
"""CLM, the decision model's client (mcp/clm.py, mcp/clm_heads.py). No GPU,
no network beyond a loopback fake server on an ephemeral port.

    python mcp/test_clm.py      -> "N/M checks passed"

WHAT THIS IS GATING (docs/CLM.md):

  1. THE HEADS: the numpy heads equal the official torch make_head
     (clm_heads.parity, copied verbatim from Contrastive-LM/CLM @ bb42c6c5)
     on a synthetic checkpoint; extract() writes a byte-deterministic .npz;
     the scale is exp(logit_scale) clamped at 100.
  2. THE STATE: fit_state never reorders the caller's evidence, keeps the
     instructions last and whole, drops whole pieces from the end the caller
     marks least important, then cuts the boundary piece, and fits CAP.
  3. ONE STATE ENCODING: decide_many encodes the state once for any number
     of questions; option vectors are cached by text hash under the encoder
     id, persisted, and a different encoder id starts empty; probabilities
     are a softmax over each option set and permute with the options.
  4. THE WIRE: HttpEncoder sends token ids and the llama-swap model name,
     reads rows by index, and every failure is ClmUnavailable with a
     retryable fact and a remedy; a held lane is CLM_BUSY, not a hang.
  5. THE REAL ARTEFACTS, when present: the extracted v0.1 heads carry the
     pinned source sha256 and scale 100; the pinned tokenizer adds no
     special tokens.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import sys
import tempfile
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_clm_")
os.environ["YAMADORI_GPU_ROOM"] = "0"
os.environ["YAMADORI_CLM_ACTIONS"] = os.path.join(_TMP, "clm_actions.npz")
os.environ["LLAMA_STACK_URL"] = "http://127.0.0.1:9"

import numpy as np  # noqa: E402

import clm  # noqa: E402
import clm_heads  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------------ fixtures --
HID = clm_heads.HIDDEN


def synthetic_npz(path: str, width: int = 32, proj: int = 16, seed: int = 0,
                  logit_scale: float = 4.6132) -> None:
    rng = np.random.default_rng(seed)
    arrays = []
    for side in ("state", "action"):
        for n in clm_heads._names(3):
            if n == "inp.weight":
                a = rng.standard_normal((width, HID)) * 0.05
            elif n.startswith("hidden") and n.endswith("weight"):
                a = rng.standard_normal((width, width)) * 0.2
            elif n == "out.weight":
                a = rng.standard_normal((proj, width)) * 0.2
            elif n.startswith("norms") and n.endswith("weight"):
                a = 1 + 0.1 * rng.standard_normal(width)
            elif n == "out.bias":
                a = 0.1 * rng.standard_normal(proj)
            else:
                a = 0.1 * rng.standard_normal(width)
            arrays.append((f"{side}.{n}", a.astype(np.float32)))
    meta = {"name": "synthetic", "cfg": {"width": width, "depth": 3,
                                         "activation": "gelu",
                                         "layernorm": True, "residual": False},
            "projection_dim": proj, "hidden_size": HID,
            "logit_scale": logit_scale,
            "scale": float(min(np.exp(logit_scale), 100.0)),
            "weights_sha256": clm_heads.weights_sha256(arrays)}
    blob = json.dumps(meta, sort_keys=True).encode()
    clm_heads.write_npz(path, arrays + [("meta", np.frombuffer(blob, np.uint8))])


class FakeTok:
    """Whitespace words -> stable ids; '\\n\\n' is its own token."""

    def ids(self, text: str) -> list[int]:
        out = []
        for para_i, para in enumerate((text or "").split("\n\n")):
            if para_i:
                out.append(1)
            out += [int(hashlib.sha1(w.encode()).hexdigest()[:6], 16) + 10
                    for w in para.split()]
        return out or [2]

    def count(self, text: str) -> int:
        return len(self.ids(text))


class FakeEncoder:
    """ids -> a deterministic unit vector; counts what it was asked."""

    def __init__(self):
        self.calls: list[int] = []

    def embed_ids(self, id_lists):
        self.calls.append(len(id_lists))
        out = []
        for ids in id_lists:
            h = hashlib.sha256(json.dumps(list(ids)).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            out.append(rng.standard_normal(HID).astype(np.float32))
        return clm_heads.l2(np.stack(out))


def fresh(heads_path: str, cache_name: str = "a.npz", encoder_id="enc-A"):
    enc = FakeEncoder()
    clm.set_encoder(enc)
    clm._tok = FakeTok()
    clm.set_action_cache(clm.ActionCache(os.path.join(_TMP, cache_name),
                                         encoder_id=encoder_id))
    return enc, clm_heads.Heads(heads_path)


# --------------------------------------------------------------- tests --
def test_heads_match_the_official_torch_modules():
    try:
        import torch
    except ImportError:
        check(False, "heads: torch is importable for the parity check")
        return
    rng = np.random.default_rng(1)
    width, proj = 48, 24
    sd = {}
    for side in ("state_head", "action_head"):
        d = {}
        for n in clm_heads._names(3):
            shape = {"inp.weight": (width, HID), "out.weight": (proj, width),
                     "out.bias": (proj,)}.get(n)
            if shape is None:
                shape = (width, width) if n.startswith("hidden") and \
                    n.endswith("weight") else (width,)
            d[n] = torch.from_numpy((rng.standard_normal(shape) * 0.1)
                                    .astype(np.float32))
        sd[side] = d
    pt = os.path.join(_TMP, "synthetic.pt")
    torch.save({**sd, "logit_scale": torch.tensor(4.9),
                "cfg": {"width": width, "depth": 3, "activation": "gelu",
                        "layernorm": True, "residual": False,
                        "projection_dim": proj, "hidden_size": HID},
                "hidden_size": HID, "projection_dim": proj}, pt)
    out = os.path.join(_TMP, "synthetic.npz")
    r = clm_heads.extract(pt, out, name="synthetic")
    check(abs(r["scale"] - 100.0) < 1e-9,
          "heads: scale = exp(logit_scale) clamped at 100 (exp 4.9 = 134)",
          str(r["scale"]))
    p = clm_heads.parity(pt, out, n=16)
    check(p["max_abs_logit"] < 1e-3 and p["argmax_same"],
          "heads: numpy == official torch make_head (logits)", json.dumps(p))
    check(p["max_abs_state"] < 1e-5 and p["max_abs_action"] < 1e-5,
          "heads: numpy == official torch make_head (unit projections)",
          json.dumps(p))
    out2 = os.path.join(_TMP, "synthetic2.npz")
    r2 = clm_heads.extract(pt, out2, name="synthetic")
    check(r["sha256"] == r2["sha256"],
          "heads: extract() writes byte-identical files", r["sha256"][:16])


def test_fit_state_keeps_order_and_the_question():
    tok = FakeTok()
    ids, info = clm.fit_state("a b c", "which one?", tok=tok, cap=50)
    check(not info["truncated"] and ids == tok.ids("a b c\n\nwhich one?"),
          "state: a fitting state is state + blank line + instructions")
    pieces = ["p1 " * 10, "p2 " * 10, "p3 " * 10]
    q = "question here"
    ids, info = clm.fit_state(pieces, q, keep="tail", tok=tok, cap=30)
    want = tok.ids("\n\n".join([pieces[1].strip(), pieces[2].strip()]) + "\n\n" + q)
    check(len(ids) <= 30 and ids[-2:] == tok.ids(q)
          and info["dropped_pieces"] >= 1,
          "state: keep=tail drops the FRONT, the question stays last and whole",
          json.dumps(info))
    check(ids == want[-len(ids):] and tok.ids("p2")[0] in ids
          and tok.ids("p1")[0] not in ids,
          "state: keep=tail keeps the later pieces in the caller's order")
    ids, info = clm.fit_state(pieces, q, keep="head", tok=tok, cap=30)
    check(len(ids) <= 30 and ids[-2:] == tok.ids(q)
          and tok.ids("p1")[0] in ids and tok.ids("p3")[0] not in ids,
          "state: keep=head drops the BACK (a most-important-first list)",
          json.dumps(info))
    ids, info = clm.fit_state(["w " * 100], q, keep="tail", tok=tok, cap=20)
    check(len(ids) == 20 and info["cut_tokens"] > 0 and ids[-2:] == tok.ids(q),
          "state: one oversized piece is cut at token level, question kept",
          json.dumps(info))
    try:
        clm.fit_state("x", "q " * 40, tok=tok, cap=10)
        check(False, "state: instructions longer than the window raise")
    except ValueError:
        check(True, "state: instructions longer than the window raise")
    check(clm.state_text(["  a  ", "", "b"], " Q ") == "a\n\nb\n\nQ",
          "state: text is the evidence joined by blank lines, question last "
          "(schema.state_text)")


def test_one_state_encoding_and_the_action_cache():
    hp = os.path.join(_TMP, "heads_small.npz")
    synthetic_npz(hp)
    enc, heads = fresh(hp)
    opts1 = ["use when writing rust", "use when writing sql", "none"]
    opts2 = ["alpha", "beta"]
    d = clm.decide_detail_many("a request", [opts1, opts2], instructions="Q?",
                               heads=heads)
    state_calls = [n for n in enc.calls if n == 1]
    check(len(d["probabilities"]) == 2 and len(d["probabilities"][0]) == 3
          and len(d["probabilities"][1]) == 2,
          "decide_many: one probability list per question")
    check(enc.calls[0] == 1 and len(state_calls) >= 1
          and d["options"]["encoded"] == 5,
          "decide_many: the state is encoded ONCE, then the 5 options",
          json.dumps({"calls": enc.calls, "options": d["options"]}))
    check(all(abs(sum(p) - 1) < 1e-9 for p in d["probabilities"]),
          "decide: a softmax over each option set")
    z = heads.project_states(enc.embed_ids([clm.fit_state(
        "a request", "Q?")[0]]))[0]
    za = heads.project_actions(enc.embed_ids([clm._tok.ids(t) for t in opts1]))
    lg = heads.logits(z, za)
    check(np.allclose(d["logits"][0], lg, atol=1e-5),
          "decide: logits are scale * cos(state_head, action_head)")
    enc.calls.clear()
    d2 = clm.decide_detail("another request", opts1, instructions="Q?",
                           heads=heads)
    check(enc.calls == [1] and d2["options"]["encoded"] == 0,
          "cache: a repeat option set costs only the state encoding",
          json.dumps({"calls": enc.calls, "options": d2["options"]}))
    perm = [2, 0, 1]
    p_a = clm.decide("same state", opts1, heads=heads)
    p_b = clm.decide("same state", [opts1[i] for i in perm], heads=heads)
    check(np.allclose([p_a[i] for i in perm], p_b, atol=1e-9),
          "decide: order-invariant (each option embedded alone)")
    # Persisted: a new process (a new ActionCache) finds the vectors.
    c2 = clm.ActionCache(os.path.join(_TMP, "a.npz"), encoder_id="enc-A")
    check(len(c2) >= 5, "cache: option vectors persist to the .npz",
          str(len(c2)))
    c3 = clm.ActionCache(os.path.join(_TMP, "a.npz"), encoder_id="enc-B")
    check(len(c3) == 0, "cache: another encoder id starts empty")
    k = clm.text_key(opts1[0])
    check(k in c2.mem and np.allclose(
        c2.mem[k], enc.embed_ids([clm._tok.ids(opts1[0])])[0], atol=1e-6),
        "cache: keyed by sha256 of the option text")
    enc2, _ = fresh(hp, cache_name="b.npz")
    info = clm.warm_options(opts1 + opts2)
    check(info["encoded"] == 5 and enc2.calls,
          "warm_options: embeds option texts ahead of use", json.dumps(info))


def test_the_wire():
    seen: list[dict] = []
    mode = {"status": 200, "width": HID}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if mode["status"] != 200:
                self.send_response(mode["status"])
                self.end_headers()
                self.wfile.write(b'{"error": "boom"}')
                return
            rows = [{"index": i, "embedding": [0.5] * mode["width"]}
                    for i in reversed(range(len(body["input"])))]
            out = json.dumps({"data": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/embeddings"
    try:
        enc = clm.HttpEncoder(url=url)
        v = enc.embed_ids([[1, 2, 3], [4]])
        check(v.shape == (2, HID) and np.allclose(np.linalg.norm(v, axis=1), 1),
              "wire: rows read by index, L2-normalised")
        b = seen[-1]
        check(b.get("model") == "clm-encoder" and b["input"] == [[1, 2, 3], [4]],
              "wire: token ids and the llama-swap model name are sent",
              json.dumps(b)[:200])
        mode["status"] = 500
        try:
            enc.embed_ids([[1]])
            check(False, "wire: a 500 is ClmUnavailable(retryable)")
        except clm.ClmUnavailable as e:
            check(e.retryable and e.code == "CLM_HTTP_ERROR" and e.remedy,
                  "wire: a 500 is ClmUnavailable(retryable) with a remedy",
                  json.dumps(e.facts()))
        mode.update(status=200, width=17)
        try:
            enc.embed_ids([[1]])
            check(False, "wire: a wrong width is refused")
        except clm.ClmUnavailable as e:
            check(e.code == "CLM_BAD_RESPONSE" and not e.retryable,
                  "wire: a wrong width is refused, not retryable")
        mode["width"] = HID
        dead = clm.HttpEncoder(url="http://127.0.0.1:9/v1/embeddings", timeout=2)
        try:
            dead.embed_ids([[1]])
            check(False, "wire: unreachable is ClmUnavailable")
        except clm.ClmUnavailable as e:
            check(e.code == "CLM_UNREACHABLE" and e.retryable,
                  "wire: unreachable is ClmUnavailable(retryable)")
        old = clm.LANE_WAIT_S
        clm.LANE_WAIT_S = 0.2
        clm._LANE.acquire()
        try:
            t0 = time.time()
            enc.embed_ids([[1]])
            check(False, "lane: a held lane is CLM_BUSY")
        except clm.ClmUnavailable as e:
            check(e.code == "CLM_BUSY" and e.retryable
                  and time.time() - t0 < 2,
                  "lane: a held lane is CLM_BUSY after LANE_WAIT_S, not a hang")
        finally:
            clm._LANE.release()
            clm.LANE_WAIT_S = old
    finally:
        srv.shutdown()


def test_the_real_artefacts_when_present():
    p = clm_heads.DEFAULT_PATH
    if os.path.exists(p):
        h = clm_heads.Heads(p)
        check(h.meta["source"]["sha256"] == clm_heads.PT_SHA256
              and h.scale == 100.0 and h.proj_dim == 512
              and h.cfg["width"] == 1536 and h.cfg["depth"] == 3,
              "artefact: v0.1 heads carry the pinned source and scale 100",
              json.dumps({"scale": h.scale, "cfg": h.cfg}))
        check(h.weights_sha256 == clm_heads.weights_sha256(
            [(f"{s}.{n}", (h.state if s == "state" else h.action).a[n])
             for s in ("state", "action") for n in clm_heads._names(3)]),
            "artefact: the heads' weights hash matches their arrays")
    else:
        check(True, f"artefact: heads not extracted here ({p}); skipped")
    if os.path.exists(clm.TOKENIZER_PATH):
        clm._tok = None
        t = clm.tokenizer()
        ids = t.ids("Hello world")
        check(ids and 151643 not in ids and 151645 not in ids and len(ids) <= 3,
              "artefact: the pinned tokenizer adds no BOS/EOS", str(ids))
        check(t.ids("") == t.ids(" "),
              "artefact: an empty text is embedded as ' ' (embed_utils.text_ids)")
    else:
        check(True, "artefact: tokenizer not present here; skipped")


def main() -> int:
    for fn in (test_heads_match_the_official_torch_modules,
               test_fit_state_keeps_order_and_the_question,
               test_one_state_encoding_and_the_action_cache,
               test_the_wire,
               test_the_real_artefacts_when_present):
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} raised",
                  traceback.format_exc().strip().split("\n")[-1])
    for ok, name, detail in _results:
        print(("  pass  " if ok else "  FAIL  ") + name
              + (f"\n        <- {detail[:300]}" if detail and not ok else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
