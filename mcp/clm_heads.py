#!/usr/bin/env python
"""CLM's projection heads, in numpy: a 4096-d encoder embedding in, the
state / action projections out, and the score of a (state, option) pair.

WHAT CLM IS (docs/CLM-EVAL.md, docs/CLM.md)

A frozen Qwen3-8B used as an embedder (last-token pooling), with two small
MLP heads on top, trained with a bidirectional InfoNCE loss. A state and each
candidate are embedded SEPARATELY; the answer to a question is a softmax over
`scale * cos(state_head(s), action_head(c))`.

THE OFFICIAL CODE THIS REPRODUCES (read only, pinned)

  https://github.com/Contrastive-LM/CLM @ bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
  (Apache-2.0)
  src/clm/heads.py   make_head(): hidden -> width -> ... -> proj;
                     forward: x = act(inp(x)); for each hidden block
                     h = act(norm(lin(x))), x = x + h if residual else h;
                     return out(x).  nn.GELU() is the exact (erf) GELU,
                     nn.LayerNorm the default eps 1e-5.
                     HeadPair._load(): scale = exp(logit_scale) clamped at
                     100; _project() L2-normalises each projection.
  src/clm/embedder.py  Embedder._fetch(): every encoder embedding is
                     L2-normalised BEFORE it reaches a head (l2(), eps 1e-12).
  src/clm/engine.py  answer(): cos = z_action @ z_state; logits =
                     scale * cos / temperature; softmax per question.

The checkpoint is Contrastive-LM/CLM-v0.1-8B @ e939398d4556fcd9400c76fa8c5a513202f42b0a,
CLM_v0.1-8B.pt, sha256 b2b4a8c9...eda5 (models/manifest.yaml `clm-heads-v0.1`):
width 1536, depth 3, LayerNorm on, no residual, GELU, projection 512,
logit_scale 4.6132 (exp = 100.8, so the served scale is the clamp, 100.0).

THE FORMAT HERE

`extract()` (torch, once, on CPU) writes the two heads to a deterministic
.npz -- fixed zip timestamps, arrays in a fixed order, float32 exactly as in
the checkpoint -- plus a JSON `meta` array (cfg, scale, the source sha256
and the pinned commit). Serving needs numpy and scipy (erf) only; torch is
never imported on the request path. `parity()` checks this implementation
against the official torch modules on random inputs.

    python mcp/clm_heads.py extract --pt <models>/CLM-v0.1-8B/CLM_v0.1-8B.pt
    python mcp/clm_heads.py parity                     # needs torch (CPU)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

CODE_REPO = "https://github.com/Contrastive-LM/CLM"
CODE_COMMIT = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
HF_REPO = "Contrastive-LM/CLM-v0.1-8B"
HF_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
PT_SHA256 = "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
HF_FILE = "CLM_v0.1-8B.pt"

HIDDEN = 4096
SCALE_MAX = 100.0          # heads.py: exp(logit_scale).clamp(max=100.0)
L2_EPS = 1e-12             # embedder.py l2()
LN_EPS = 1e-5              # nn.LayerNorm default
PROJ_EPS = 1e-12           # F.normalize default eps

DEFAULT_PATH = os.environ.get(
    "YAMADORI_CLM_HEADS",
    os.path.join(ROOT, "index", "clm", "heads", "CLM_v0.1-8B.npz"))


def l2(x: np.ndarray, axis: int = -1, eps: float = L2_EPS) -> np.ndarray:
    """embedder.py's l2(): x / (||x|| + eps)."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    """nn.GELU() (approximate='none'): 0.5 x (1 + erf(x / sqrt 2))."""
    from scipy.special import erf
    return (0.5 * x * (1.0 + erf(x / np.float32(np.sqrt(2.0))))).astype(
        np.float32)


def _layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)       # biased, as torch
    return ((x - mu) / np.sqrt(var + np.float32(LN_EPS)) * w + b).astype(
        np.float32)


class Head:
    """One MLP head (make_head) over float32 numpy arrays."""

    def __init__(self, arrays: dict[str, np.ndarray], cfg: dict):
        self.a = arrays
        self.depth = int(cfg["depth"])
        self.layernorm = bool(cfg.get("layernorm", False))
        self.residual = bool(cfg.get("residual", False))
        act = cfg.get("activation", "gelu")
        if act != "gelu":
            raise ValueError(f"activation {act!r}: only gelu is implemented "
                             f"(the v0.1 checkpoint's)")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        a = self.a
        x = np.asarray(x, dtype=np.float32)
        x = _gelu(x @ a["inp.weight"].T + a["inp.bias"])
        for i in range(self.depth - 2):
            h = x @ a[f"hidden.{i}.weight"].T + a[f"hidden.{i}.bias"]
            if self.layernorm:
                h = _layernorm(h, a[f"norms.{i}.weight"], a[f"norms.{i}.bias"])
            h = _gelu(h)
            x = x + h if self.residual else h
        return (x @ a["out.weight"].T + a["out.bias"]).astype(np.float32)


class Heads:
    """The state head, the action head and the scale, from one .npz."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            arrays = {k: np.array(z[k]) for k in z.files if k != "meta"}
        self.meta = meta
        cfg = meta["cfg"]
        self.cfg = cfg
        self.scale = float(meta["scale"])
        self.proj_dim = int(meta["projection_dim"])
        self.state = Head({k[len("state."):]: v for k, v in arrays.items()
                           if k.startswith("state.")}, cfg)
        self.action = Head({k[len("action."):]: v for k, v in arrays.items()
                            if k.startswith("action.")}, cfg)
        self.weights_sha256 = meta.get("weights_sha256")
        # Identity of these exact weights, for cache keys of projections.
        self.namespace = f"{meta.get('name', 'clm')}@{(self.weights_sha256 or '')[:12]}"

    def project_states(self, emb: np.ndarray) -> np.ndarray:
        """[n, 4096] encoder embeddings (normalised here, as embedder.py
        does) -> [n, proj] unit projections."""
        return l2(self.state(l2(np.atleast_2d(emb))), eps=PROJ_EPS)

    def project_actions(self, emb: np.ndarray) -> np.ndarray:
        return l2(self.action(l2(np.atleast_2d(emb))), eps=PROJ_EPS)

    def logits(self, z_state: np.ndarray, z_actions: np.ndarray,
               temperature: float = 1.0) -> np.ndarray:
        """engine.py: scale * (z_actions @ z_state) / temperature."""
        return (self.scale * (np.atleast_2d(z_actions) @ np.asarray(
            z_state, dtype=np.float32).reshape(-1)) / float(temperature)
                ).astype(np.float64)


def softmax(logits) -> np.ndarray:
    v = np.asarray(logits, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


_loaded: dict[str, Heads] = {}
_lock = threading.Lock()


def load(path: str | None = None) -> Heads:
    """The heads at `path` (default DEFAULT_PATH), loaded once per process."""
    p = os.path.abspath(path or DEFAULT_PATH)
    with _lock:
        h = _loaded.get(p)
        if h is None:
            h = _loaded[p] = Heads(p)
        return h


# ------------------------------------------------------------ extraction --
def _names(depth: int) -> list[str]:
    """make_head's state-dict keys, in the order they are written."""
    order = ["inp.weight", "inp.bias"]
    for i in range(depth - 2):
        order += [f"hidden.{i}.weight", f"hidden.{i}.bias"]
    for i in range(depth - 2):
        order += [f"norms.{i}.weight", f"norms.{i}.bias"]
    return order + ["out.weight", "out.bias"]


def weights_sha256(arrays: list[tuple[str, np.ndarray]]) -> str:
    h = hashlib.sha256()
    for name, a in arrays:
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(repr(tuple(a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def write_npz(path: str, arrays: list[tuple[str, np.ndarray]]) -> None:
    """A deterministic .npz: fixed order, fixed zip timestamps, stored."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, a in arrays:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(a),
                                      allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())
    os.replace(tmp, path)


def extract(pt: str, out: str = DEFAULT_PATH, name: str = "CLM_v0.1-8B") -> dict:
    """Read the official .pt (torch, CPU, weights_only) and write the .npz."""
    import torch
    with open(pt, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    ck = torch.load(pt, map_location="cpu", weights_only=True)
    cfg = dict(ck["cfg"])
    proj = int(ck.get("projection_dim", cfg.get("projection_dim", 512)))
    hidden = int(ck.get("hidden_size", cfg.get("hidden_size", HIDDEN)))
    raw_scale = float(torch.as_tensor(ck["logit_scale"]).float())
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(
        max=SCALE_MAX))
    arrays: list[tuple[str, np.ndarray]] = []
    for side in ("state", "action"):
        sd = ck[f"{side}_head"]
        names = _names(int(cfg["depth"]))
        if sorted(names) != sorted(sd.keys()):
            raise ValueError(f"{side}_head keys {sorted(sd)} do not match "
                             f"make_head(depth={cfg['depth']}): {names}")
        for n in names:
            arrays.append((f"{side}.{n}",
                           sd[n].detach().cpu().numpy().astype(np.float32)))
    wsha = weights_sha256(arrays)
    meta = {"name": name, "cfg": cfg, "projection_dim": proj,
            "hidden_size": hidden, "logit_scale": raw_scale, "scale": scale,
            "source": {"file": os.path.basename(pt), "sha256": sha,
                       "hf_repo": HF_REPO, "hf_revision": HF_REVISION},
            "code": {"repo": CODE_REPO, "commit": CODE_COMMIT,
                     "files": ["src/clm/heads.py", "src/clm/embedder.py",
                               "src/clm/engine.py", "src/clm/schema.py"]},
            "weights_sha256": wsha}
    if os.path.basename(pt) == HF_FILE and sha != PT_SHA256:
        meta["warning"] = f"source sha256 {sha} is not the pinned {PT_SHA256}"
    elif os.path.basename(pt) != HF_FILE:
        # A fine-tuned head (train/finetune.py's best_head.pt): not the
        # release; its source is recorded by hash, not by the HF pin.
        meta["source"].update(hf_repo=None, hf_revision=None,
                              derived_from=HF_FILE)
    blob = json.dumps(meta, sort_keys=True).encode("utf-8")
    write_npz(out, arrays + [("meta", np.frombuffer(blob, dtype=np.uint8))])
    with open(out, "rb") as fh:
        file_sha = hashlib.sha256(fh.read()).hexdigest()
    return {"out": out, "size": os.path.getsize(out), "sha256": file_sha,
            "weights_sha256": wsha, "scale": scale, "logit_scale": raw_scale,
            "cfg": cfg, "source_sha256": sha}


def parity(pt: str, path: str = DEFAULT_PATH, n: int = 64,
           seed: int = 0) -> dict:
    """This numpy implementation vs the official torch modules (make_head,
    copied verbatim below from the pinned commit) on `n` random unit
    vectors. Returns the max abs difference of the projections and of the
    logits."""
    import torch
    import torch.nn as nn

    def make_head(width, depth=2, proj=512, activation="gelu",
                  layernorm=False, residual=False, hidden=HIDDEN):
        # src/clm/heads.py @ CODE_COMMIT, verbatim apart from indentation.
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(hidden, width)
                self.hidden = nn.ModuleList(nn.Linear(width, width)
                                            for _ in range(depth - 2))
                self.norms = nn.ModuleList(
                    (nn.LayerNorm(width) if layernorm else nn.Identity())
                    for _ in range(depth - 2))
                self.out = nn.Linear(width, proj)
                self.act = act()
                self.residual = residual

            def forward(self, x):
                x = self.act(self.inp(x))
                for lin, nrm in zip(self.hidden, self.norms):
                    h = self.act(nrm(lin(x)))
                    x = x + h if self.residual else h
                return self.out(x)
        return Head()

    ck = torch.load(pt, map_location="cpu", weights_only=True)
    cfg = dict(ck["cfg"])
    kw = dict(width=cfg["width"], depth=cfg["depth"],
              proj=ck.get("projection_dim", cfg.get("projection_dim", 512)),
              activation=cfg.get("activation", "gelu"),
              layernorm=cfg.get("layernorm", False),
              residual=cfg.get("residual", False),
              hidden=cfg.get("hidden_size", HIDDEN))
    sh, ah = make_head(**kw), make_head(**kw)
    sh.load_state_dict(ck["state_head"])
    ah.load_state_dict(ck["action_head"])
    sh.eval()
    ah.eval()
    scale_t = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(
        max=SCALE_MAX))
    rng = np.random.default_rng(seed)
    x = l2(rng.standard_normal((n, HIDDEN)).astype(np.float32))
    with torch.no_grad():
        zs_t = torch.nn.functional.normalize(sh(torch.from_numpy(x)),
                                             dim=-1).numpy()
        za_t = torch.nn.functional.normalize(ah(torch.from_numpy(x)),
                                             dim=-1).numpy()
    h = Heads(path)
    zs, za = h.project_states(x), h.project_actions(x)
    lg_t = scale_t * (za_t @ zs_t[0])
    lg = h.logits(zs[0], za)
    return {"n": n, "scale_torch": scale_t, "scale_npz": h.scale,
            "max_abs_state": float(np.abs(zs - zs_t).max()),
            "max_abs_action": float(np.abs(za - za_t).max()),
            "max_abs_logit": float(np.abs(lg - lg_t).max()),
            "argmax_same": bool(int(np.argmax(lg)) == int(np.argmax(lg_t)))}


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--pt", required=True)
    e.add_argument("--out", default=DEFAULT_PATH)
    p = sub.add_parser("parity")
    p.add_argument("--pt", required=True)
    p.add_argument("--path", default=DEFAULT_PATH)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        print(json.dumps(extract(a.pt, a.out), indent=1))
    else:
        r = parity(a.pt, a.path)
        print(json.dumps(r, indent=1))
        return 0 if (r["max_abs_logit"] < 1e-3 and r["argmax_same"]) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
