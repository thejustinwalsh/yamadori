#!/usr/bin/env python
"""THE SERVER TOOLBOX: `yama_compose_tools` (docs/CODEMODE.md).

Operator, 2026-10-07: "pi now has codemode ... we could easily structure our proxy to allow the server to facilitate
this feature ... we should make codemode in our server. QuickJS sandbox is a good idea, I love this." And: "yama_run_code
is a terrible name, a part of code mode are the calls to find and describe tools ... let's be thoughtful of the feature
and not just think it's running a JS script in the server." And: "We can use yama_prefix for server tool identity here
too."

DISCOVERY + COMPOSITION. The model writes a short JavaScript program that FINDS the server's tools
(`tools.search`, `tools.describe`, `tools.namespaces`, `tools.all`) and CALLS several together
(`tools.yama_find_package({...})`, a Promise, at most four at a time); the QuickJS runner is only the mechanism.

This module is the LIBRARY and has no import of the proxy:

  Toolbox        the registry as data (functions, namespaces): BM25 search, describe, namespaces, all
  build_toolbox  the registry from what a conversation was offered (reads each tool's own definition: one source)
  tool_definition the one tool main is sent: trigger-condition description + TypeScript declarations under a budget
  run            the sandbox: QuickJS on WASI under wasmtime (mcp/codemode_sandbox.py), this module bridging calls
  format_result  "Script completed" / "Script failed", wall time, text items, <console_output>, capped

The proxy (proxy.run_compose_tools) supplies `call(name, args, seq) -> ToolResult` and owns the accounting.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

TOOL_NAME = "yama_compose_tools"
# THE HARNESS CODE-MODE TOOLS (operator, 2026-10-08, verbatim: "if we detect pi or other harnesses codemode in the
# tools list we do not turn on internal codemode, it is meant as a fallback for other harnesses that don't already
# have harness level codemode."): a client that brings one gets no yama_compose_tools. ONE constant; a name goes in
# only from evidence (a harness fixture or a recorded request), never a guess. Pi: https://pi.dev/docs/latest/codemode
# -- "the model's tool list holds one tool named `codemode`" (read 2026-10-08). proxy.TOOL_OVERLAPS reads it.
HARNESS_TOOLS = ("codemode",)
HARNESS_OVERLAP_WHY = "the client brings its own code mode"
SWITCH = "codemode"
MODES = ("on", "only")
DEFAULT_MODE = "on"
MODE_ENV = "YAMADORI_CODEMODE_MODE"

# ---------------------------------------------------------------- limits ---
# Pi's (https://pi.dev/docs/latest/codemode, read 2026-10-07): "256 MB of memory"; "At most four such calls run at
# once per script"; "Scripts fail when output exceeds 16777216 characters ... or 100000 text(), image(), and
# console calls"; searchTools' "default limit 8"; codemode.inlineBudget "3000" tokens.
MEMORY_BYTES = 256 * 1024 * 1024
CONCURRENCY = 4
MAX_OUTPUT_CHARS = 16_777_216
MAX_OUTPUT_CALLS = 100_000
SEARCH_LIMIT = 8
INLINE_BUDGET_CEILING_TOKENS = 3000
# What a script's result may add to the conversation: Pi's default output limit is 10,000 tokens ("longer output is
# truncated"); at the repo's ~3 characters a token (tokens_of) that is 30,000 characters. Operator, 2026-10-08:
# "we should increase output limit" (it was repeats.RESULT_CAP, 6,000, the breaker of every other tool result).
OUTPUT_CAP = 30_000
# Pi's store(): "262,144 characters max per value and 1,048,576 total", writes persist only if the script succeeds.
STORE_VALUE_CHARS = 262_144
STORE_TOTAL_CHARS = 1_048_576
STORE_CONVERSATIONS = 256                     # conversations whose store is kept in memory (least recently used out)
MAX_IMAGES = 8                                # image() calls shown per script
MAX_IMAGE_BYTES = 12 * 1024 * 1024

_STORES: "OrderedDict[str, dict]" = OrderedDict()
_STORES_LOCK = threading.Lock()


def store_for(key: str) -> dict:
    """The conversation's store: {name: JSON text}. In memory (a restart forgets it), least recently used
    conversation out past STORE_CONVERSATIONS; an empty key (a side call) gets a throwaway dict."""
    if not key:
        return {}
    with _STORES_LOCK:
        s = _STORES.get(key)
        if s is None:
            s = _STORES[key] = {}
            while len(_STORES) > STORE_CONVERSATIONS:
                _STORES.popitem(last=False)
        else:
            _STORES.move_to_end(key)
        return s


def image_bytes(v) -> tuple[bytes | None, str]:
    """(PNG bytes, "") for an image() argument -- a base64 data URL, an { image_url } (a string or { url }) or an
    image block { type: "image", data | source: { data } } -- else (None, why). Remote URLs are not fetched (Pi's
    rule, and the sandbox has no network). Any raster format Pillow reads becomes a PNG."""
    import base64
    import io
    url = None
    if isinstance(v, str):
        url = v
    elif isinstance(v, dict):
        iu = v.get("image_url")
        url = iu.get("url") if isinstance(iu, dict) else iu
        if not url:
            src = v.get("source") if isinstance(v.get("source"), dict) else v
            data = src.get("data")
            if isinstance(data, str):
                url = data if data.startswith("data:") else f"data:{src.get('mimeType') or src.get('media_type') or 'image/png'};base64,{data}"
    m = re.match(r"^data:([\w./+-]*);base64,(.*)$", url or "", re.S)
    if not m:
        return None, "image() takes a base64 data URL, { image_url } or an image block; remote URLs are not fetched"
    try:
        raw = base64.b64decode(m.group(2), validate=False)
    except (ValueError, TypeError):
        return None, "the image data is not valid base64"
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return None, f"the image is empty or larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return raw, ""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        im.load()
        out = io.BytesIO()
        im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(out, format="PNG")
        return out.getvalue(), ""
    except Exception:                                            # noqa: BLE001
        return None, "the bytes are not an image the server can read"
# The decider's own per-read timeout (decider_bonsai.read/ask_one `timeout: float = 120`; test_codemode checks it
# against the signature): one typed question is one read; craft's question path makes two (the choice and the gate).
DECIDER_READ_S = 120.0
CRAFT_BOUND_S = 2 * DECIDER_READ_S
DECIDE_BOUND_S = DECIDER_READ_S

_STOP = frozenset("a an and are as at be by do does for from how i in is it of on or that the this to what which with "
                  "you your can get use used using into than then there their them they not no if when where who "
                  "whom why will would should could has have had was were been being its it's about".split())


# ---------------------------------------------------------------- text -----
def words(text: str) -> list[str]:
    """Lower-case words of a text, a plural's s dropped (BM25's tokens). Underscores split: yama_find_package is
    yama, find, package."""
    out = []
    for w in re.findall(r"[a-z0-9]+", str(text or "").lower()):
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in _STOP and w != "yama" and len(w) > 1:
            out.append(w)
    return out


def tokens_of(text: str) -> int:
    """The proxy's own high estimate (tiers.estimate_prompt_tokens: ~3 characters a token)."""
    return len(text or "") // 3


def split_description(desc: str) -> tuple[str, str]:
    """(the question a tool answers, the rest -- its trigger conditions) from a tool description in the house form
    "Answers '<question>?': <what it returns>. Call it when ... A server tool: ..." (AGENTS.md "Tool descriptions
    are prompts"); any other description is cut at its first sentence."""
    d = " ".join(str(desc or "").split())
    m = re.match(r"Answers (?:a question about )?'([^']+)'[:.]?\s*(.*)$", d)
    if m:
        q, rest = m.group(1).strip(), m.group(2)
    else:
        m = re.match(r"(.+?[.!?])(?:\s+(.*))?$", d)
        q, rest = (m.group(1), m.group(2) or "") if m else (d, "")
    rest = re.sub(r"\s*A server tool: it runs on the Yamadori server and does not touch your workspace\.?\s*$", "",
                  rest)
    return q[:200], rest


# ------------------------------------------------------- JSON schema -> TS --
def ts_type(schema, comments: bool = False, indent: int = 0) -> str:
    """A TypeScript type for a JSON Schema (the subset our tool definitions use)."""
    if not isinstance(schema, dict):
        return "any"
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return " | ".join(json.dumps(v) for v in schema["enum"])
    t = schema.get("type")
    if isinstance(t, list):
        return " | ".join(ts_type(dict(schema, type=x), comments, indent) for x in t)
    if t in ("integer", "number"):
        return "number"
    if t in ("string", "boolean", "null"):
        return t
    if t == "array":
        inner = ts_type(schema.get("items"), comments, indent)
        return f"({inner})[]" if " | " in inner else f"{inner}[]"
    if t == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        req = set(schema.get("required") or [])
        if not props:
            return "Record<string, any>"
        pad = "  " * (indent + 1)
        parts = []
        for k, v in props.items():
            doc = ""
            if comments and isinstance(v, dict) and v.get("description"):
                doc = f"{pad}/** {' '.join(str(v['description']).split())} */\n"
            parts.append(f"{doc}{pad}{k}{'' if k in req else '?'}: {ts_type(v, comments, indent + 1)};")
        if comments:
            return "{\n" + "\n".join(parts) + "\n" + "  " * indent + "}"
        return "{ " + " ".join(p.strip() for p in parts) + " }"
    return "any"


def example_value(schema, name: str = "") -> object:
    """A placeholder value for a schema: the first enum member, an empty string/list, 1, false."""
    if not isinstance(schema, dict):
        return None
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "string":
        return "..."
    if t in ("integer", "number"):
        return 1
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {k: example_value(v, k) for k, v in (schema.get("properties") or {}).items()
                if k in (schema.get("required") or [])}
    return None


def example_args(params: dict) -> str:
    props = (params or {}).get("properties") or {}
    req = list((params or {}).get("required") or [])
    return json.dumps({k: example_value(props.get(k), k) for k in req})


# ---------------------------------------------------------------- Toolbox ---
class Toolbox:
    """The registry as DATA: functions (each a tool of ours) in namespaces. Built by `build_toolbox` in the proxy
    from the tools the conversation was offered; rebuilt from `catalog()` in the sandbox's child process."""

    def __init__(self, fns: list[dict], namespaces: dict[str, dict] | None = None):
        self.fns = list(fns)
        self.by_name = {f["name"]: f for f in self.fns}
        order = []
        for f in self.fns:
            if f["ns"] not in order:
                order.append(f["ns"])
        ns = namespaces or {}
        self.namespaces = {n: dict(ns.get(n) or {"description": n}) for n in order}
        self._index = None

    # -- data
    def catalog(self) -> dict:
        return {"fns": self.fns, "namespaces": self.namespaces}

    @classmethod
    def from_catalog(cls, cat: dict) -> "Toolbox":
        return cls(cat.get("fns") or [], cat.get("namespaces") or {})

    def names(self) -> list[str]:
        return [f["name"] for f in self.fns]

    def bound_s(self) -> float:
        """The wall ceiling of a script: the slowest exposed function's own bound (docs/CODEMODE.md)."""
        return max([float(f.get("bound_s") or 0) for f in self.fns] or [0.0])

    # -- BM25 over name + description + trigger text
    def _doc_words(self, f: dict) -> list[str]:
        return (words(f["name"]) * 3 + words(f.get("question")) * 2 + words(f.get("when"))
                + words(" ".join((f.get("parameters") or {}).get("properties") or {})))

    def _bm25(self):
        if self._index is None:
            docs = {f["name"]: self._doc_words(f) for f in self.fns}
            n = max(len(docs), 1)
            df: dict[str, int] = {}
            for ws in docs.values():
                for w in set(ws):
                    df[w] = df.get(w, 0) + 1
            avg = sum(len(w) for w in docs.values()) / n if docs else 1.0
            self._index = (docs, df, n, max(avg, 1.0))
        return self._index

    def scores(self, query: str, namespace: str | None = None) -> dict[str, float]:
        """{function: BM25 score} for the functions the query shares a word with. Okapi BM25, k1 1.5, b 0.75 (the
        textbook defaults; deterministic: no embedder, ties by registry order)."""
        docs, df, n, avg = self._bm25()
        k1, b = 1.5, 0.75
        q = words(query)
        out: dict[str, float] = {}
        for f in self.fns:
            if namespace and f["ns"] != namespace:
                continue
            ws = docs[f["name"]]
            tf: dict[str, int] = {}
            for w in ws:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in set(q):
                if w not in tf:
                    continue
                idf = math.log(1.0 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(ws) / avg))
            if s > 0:
                out[f["name"]] = round(s, 6)
        return out

    # -- the four discovery calls (sync, in the sandbox)
    def search(self, query, opts: dict | None = None) -> list[dict]:
        opts = opts if isinstance(opts, dict) else {}
        try:
            limit = int(opts.get("limit") or SEARCH_LIMIT)
        except (TypeError, ValueError):
            limit = SEARCH_LIMIT
        ns = opts.get("namespace")
        order = {f["name"]: i for i, f in enumerate(self.fns)}
        sc = self.scores(str(query or ""), ns if isinstance(ns, str) and ns else None)
        ranked = sorted(sc, key=lambda n: (-sc[n], order[n]))[:max(limit, 1)]
        return [{"name": n, "namespace": self.by_name[n]["ns"], "description": self.by_name[n].get("question") or "",
                 "score": sc[n]} for n in ranked]

    def resolve(self, name) -> dict | None:
        """The function a name means: exact, or without its yama_ prefix."""
        n = str(name or "").strip()
        n = re.sub(r"^tools\.", "", n)
        if n in self.by_name:
            return self.by_name[n]
        if "yama_" + n in self.by_name:
            return self.by_name["yama_" + n]
        hits = [f for f in self.fns if f["name"].endswith("_" + n)]
        return hits[0] if len(hits) == 1 else None

    def describe(self, name) -> dict:
        f = self.resolve(name)
        if f is None:
            near = [h["name"] for h in self.search(str(name), {"limit": 3})]
            return {"ok": False, "error": f"no function named {str(name)[:80]!r}",
                    "near": near,
                    "next": ("tools.search(\"what you need\") ranks the functions by what they answer; tools.all() "
                             "lists every one")}
        return {"ok": True, "name": f["name"], "namespace": f["ns"], "question": f.get("question"),
                "description": f["description"],
                "declaration": declaration(f, comments=True),
                "returns": "Promise<ToolResult>: { ok, text, data?, error? }; it never throws",
                "example": f"await tools.{f['name']}({example_args(f.get('parameters') or {})})",
                "bound_s": f.get("bound_s")}

    def namespaces_info(self, name=None) -> object:
        name = str(name or "").strip()
        if name:
            ns = self.namespaces.get(name)
            if ns is None:
                return {"ok": False, "error": f"no namespace named {name[:80]!r}",
                        "namespaces": list(self.namespaces),
                        "next": "tools.namespaces() lists them"}
            return {"ok": True, "name": name, "description": ns.get("description"),
                    "instructions": ns.get("instructions"),
                    "functions": [{"name": f["name"], "description": f.get("question") or ""}
                                  for f in self.fns if f["ns"] == name]}
        return [{"name": n, "description": v.get("description"),
                 "functions": [f["name"] for f in self.fns if f["ns"] == n]}
                for n, v in self.namespaces.items()]

    def all(self) -> list[dict]:
        return [{"name": f["name"], "namespace": f["ns"], "description": f.get("question") or ""}
                for f in self.fns]


def declaration(f: dict, comments: bool = False) -> str:
    """One function as TypeScript. The comment is the question the tool answers (its own description's opening: one
    source); `comments` adds the parameters' descriptions (tools.describe)."""
    params = f.get("parameters") or {"type": "object", "properties": {}}
    q = f"/** {f.get('question') or f['name']} */\n" if f.get("question") else ""
    if comments:
        q = f"/** {' '.join(str(f['description']).split())} */\n"
    return f"{q}{f['name']}(args: {ts_type(params, comments)}): Promise<ToolResult>;"


# ------------------------------------------------------------ the switch ---
def mode_source(tier: dict | None) -> tuple[str, str]:
    """(mode, source): the header's `codemode_mode` (tier overridden), else YAMADORI_CODEMODE_MODE, else `on`."""
    tier = tier or {}
    if "codemode_mode" in (tier.get("overridden") or []):
        v = str(tier.get("codemode_mode") or "").strip().lower()
        if v in MODES:
            return v, "header"
        return DEFAULT_MODE, f"header (unknown mode {v!r})"
    v = os.environ.get(MODE_ENV, "").strip().lower()
    if v in MODES:
        return v, "env"
    return DEFAULT_MODE, "default"


def switch(tier: dict | None) -> dict:
    """{on, source, mode, mode_source}: tiers.BEHAVIOURS `codemode` (OFF BY DEFAULT until its probe passes; medium
    and up where on) and its mode."""
    import tiers
    on, src = tiers.behaviour_source(tier, SWITCH)
    mode, msrc = mode_source(tier)
    return {"on": bool(on), "source": src, "mode": mode, "mode_source": msrc}


# ------------------------------------------------------------- the tools ---
TOOL_RESULT_TYPE = ("type ToolResult = { ok: boolean; text: string; data?: any; "
                    "error?: { code: string; reason: string; retryable: boolean; next: string } };")

BUILTIN_DECLARATIONS = (
    "  /** Rank the server's functions by what they answer (BM25 over name, description and trigger text). */\n"
    "  search(query: string, opts?: { limit?: number; namespace?: string }): "
    "{ name: string; namespace: string; description: string; score: number }[];\n"
    "  /** One function in full: description, parameters, an example call. */\n"
    "  describe(name: string): { ok: boolean; name?: string; description?: string; declaration?: string; "
    "example?: string };\n"
    "  /** The groups of functions; with a name, one group's functions. */\n"
    "  namespaces(name?: string): any;\n"
    "  /** Every function: { name, namespace, description }. */\n"
    "  all(): { name: string; namespace: string; description: string }[];")

GLOBAL_DECLARATIONS = (
    "declare function text(value: any): void;                       // add an output item\n"
    "declare function image(value: string | { image_url: string }): void;  // base64 data URL; shown to the user\n"
    "declare function store(key: string, value: any): void;         // small JSON, kept for this conversation's next scripts (also save)\n"
    "declare function load(key: string): any;                        // undefined when absent\n"
    "declare function sleep(ms: number): Promise<void>;")

SERVER_TOOL_SENTENCE = ("A server tool: it runs on the Yamadori server and does not touch your workspace.")

DECIDE_NAMESPACE = {
    "description": ("Classify a text with jjava, the server's decision model: it reads the text and answers one typed "
                    "question about it (yes/no, one of several options, a score), with probabilities. Use it to "
                    "classify, filter or rank many items in code."),
}
DECIDE_STATE = {"type": "string", "description": "The text to read: the item the question is about."}
DECIDE_TOOLS = [
    {"type": "function", "function": {
        "name": "yama_classify_yes_no",
        "description": ("Answers 'is this true of this text?': jjava reads a text and returns the probability that "
                        "a yes/no question about it is yes. Call it to check, filter or sort many items in code "
                        "-- one call per item, in parallel -- instead of reading them all yourself: is this README "
                        "about X, does this error mean Y, is this file the entry point. " + SERVER_TOOL_SENTENCE),
        "parameters": {"type": "object", "properties": {
            "state": DECIDE_STATE,
            "question": {"type": "string", "description": "A yes/no question about the text."},
            "criteria": {"type": "object", "description": "Optional: what yes and no mean.",
                         "properties": {"true": {"type": "string"}, "false": {"type": "string"}}}},
            "required": ["state", "question"]}}},
    {"type": "function", "function": {
        "name": "yama_classify_choice",
        "description": ("Answers 'which of these options fits this text?': jjava reads a text and returns its "
                        "probability for each option you list, and the best one. Call it to sort items into "
                        "categories in code, one call per item, in parallel. " + SERVER_TOOL_SENTENCE),
        "parameters": {"type": "object", "properties": {
            "state": DECIDE_STATE,
            "question": {"type": "string", "description": "The question the options answer."},
            "options": {"type": "object", "description": ("The options: { key: what it means }. Up to 255.")}},
            "required": ["state", "question", "options"]}}},
    {"type": "function", "function": {
        "name": "yama_classify_score",
        "description": ("Answers 'how much does this text have of something?': jjava reads a text and returns a "
                        "score on the ordered scale you give, 0 for the first level up, with the probability of "
                        "each level. Call it to rank items in code, one call per item, in parallel. "
                        + SERVER_TOOL_SENTENCE),
        "parameters": {"type": "object", "properties": {
            "state": DECIDE_STATE,
            "question": {"type": "string", "description": "What is being scored."},
            "levels": {"type": "array", "items": {"type": "string"},
                       "description": "2 to 10 level descriptions, lowest first."}},
            "required": ["state", "question", "levels"]}}},
]
DECIDE_NAMES = tuple(t["function"]["name"] for t in DECIDE_TOOLS)


def _fn(defn: dict, ns: str, kind: str, bound_s: float | None, external: bool) -> dict:
    f = defn["function"]
    q, when = split_description(f.get("description") or "")
    return {"name": f["name"], "ns": ns, "kind": kind, "description": f.get("description") or "",
            "question": q, "when": when, "parameters": f.get("parameters") or {"type": "object", "properties": {}},
            "bound_s": bound_s, "external": bool(external)}


def build_toolbox(names: list[str], decider: bool = False) -> Toolbox:
    """The toolbox for a conversation: one function per tool of ours it was OFFERED (`names`: yama_* names; a name
    withheld for a client tool is not in it), each read from the tool's own definition -- the one source of its
    description and parameters -- plus the decide functions where jjava is on. Order: MCP servers (config order),
    craft, decide, images."""
    import images
    import mcp_config
    import skill_prompts
    import skill_select
    import vision
    want = {mcp_config.canonical(n) for n in names or []}
    fns, nss = [], {}
    for spec in mcp_config.servers():
        ns = spec.get("namespace") or spec["id"]
        for t in spec.get("tools") or []:
            if t["name"] in want:
                bound = t.get("call_timeout_s") or spec.get("call_timeout_s")
                fns.append(_fn(mcp_config.definition(t), ns, "mcp", bound, True))
                nss.setdefault(ns, {
                    "description": spec.get("namespace_description") or (
                        f"{spec.get('title') or spec['id']} ({spec.get('package') or 'an MCP server'}), "
                        f"hosted by the proxy"),
                    "instructions": spec.get("line", "").strip() or None})
    if skill_prompts.CRAFT_TOOL_NAME in want:
        fns.append(_fn(skill_select.READ_TOOL, "craft", "craft", CRAFT_BOUND_S, False))
        nss["craft"] = {"description": "This service's craft library: short, sourced moves for a library or kind "
                                       "of work."}
    if decider:
        for d in DECIDE_TOOLS:
            fns.append(_fn(d, "classify", "decide", DECIDE_BOUND_S, False))
        nss["classify"] = dict(DECIDE_NAMESPACE)
    if images.TOOL_NAME in want:
        fns.append(_fn(images.MAIN_TOOL, "images", "images", images.timeout(), False))
    if vision.TOOL_NAME in want:
        fns.append(_fn(vision.TOOL, "images", "vision", vision.timeout(), False))
    if any(f["ns"] == "images" for f in fns):
        nss["images"] = {"description": "Draw a picture from words, look at one (an attached image or one drawn "
                                        "here). A drawn picture is shown to the user by the server."}
    return Toolbox(fns, nss)


# -------------------------------------------------------- the one tool -----
SYSTEM_LINE = ("\n\nTo get several facts from the server at once, write a short script for "
               f"{TOOL_NAME}; tools.search lists what the server can do.")

_HEAD = (
    "Answers 'how do I get several facts from the server in one go?': you write a short JavaScript program (the body "
    "of an async function) that finds the server's tools and calls them together, and you read back only what it "
    "returns or prints. Call it when you need two or more lookups at once (several packages' names, versions and "
    "READMEs; a craft and a package), when you need only part of a long result (slice, filter or count it in code), "
    "or when you do not know which server function answers your question: tools.search(\"what you need\") ranks "
    "them and tools.describe(name) shows one in full. Calls return Promises and run in parallel, four at a time: use "
    "await and Promise.all. Output is what you return, text(value) and console.log write, cut at {cap} characters, "
    "so print only what you need. store(key, value) and load(key) keep small JSON for your next scripts; "
    "image(dataUrl) shows a picture to the user. The sandbox has no network or files; bash, read and write are your "
    "client's tools and cannot be called from it. " + SERVER_TOOL_SENTENCE)


def _example(tb: Toolbox, shown: list[dict]) -> str:
    pick = [f for f in shown if f.get("parameters", {}).get("required")][:2] or shown[:1]
    if not pick:
        return "return tools.all();"
    first = pick[0]
    q = (first.get("question") or first["name"]).replace('"', "'")[:48]
    calls = ", ".join(f"tools.{f['name']}({example_args(f.get('parameters') or {})})" for f in pick)
    return (f"const found = tools.search(\"{q}\", {{ limit: 3 }});\n"
            f"const results = await Promise.all([{calls}]);\n"
            "return found.map(h => h.name).join(\", \") + \"\\n\" + "
            "results.map(r => r.text.slice(0, 300)).join(\"\\n---\\n\");")


def tool_definition(tb: Toolbox, *, rank_query: str = "", budget_tokens: int | None = None,
                    cap_chars: int | None = None) -> tuple[dict, dict]:
    """(the tool main is sent, its record). The description carries the declarations of the functions most
    relevant to `rank_query` (BM25; registry order when nothing matches) that fit `budget_tokens` -- the cost of
    the whole definition, so the toolbox stays size-neutral beside the tools it covers -- and a line naming what is
    left out. Decided ONCE per conversation and stored rendered (the prefix never changes)."""
    if cap_chars is None:
        cap_chars = OUTPUT_CAP
    budget = int(budget_tokens if budget_tokens is not None else INLINE_BUDGET_CEILING_TOKENS)
    sc = tb.scores(rank_query) if rank_query else {}
    order = {f["name"]: i for i, f in enumerate(tb.fns)}
    ranked = sorted(tb.names(), key=lambda n: (-sc.get(n, 0.0), order[n]))
    schema = {"type": "object", "properties": {
        "script": {"type": "string", "description": (
            "JavaScript: the body of an async function. Use tools.<function>(args), await, return, text(), "
            "console.log.")},
        "timeout_ms": {"type": "integer", "description": (
            "Optional: stop the script after this many milliseconds (it can only shorten the server's own limit).")}},
        "required": ["script"]}

    def build(declared: list[str]) -> str:
        groups: dict[str, list[dict]] = {}
        for f in tb.fns:
            if f["name"] in declared:
                groups.setdefault(f["ns"], []).append(f)
        lines = []
        for ns, fs in groups.items():
            lines.append(f"  // {ns}: {(tb.namespaces.get(ns) or {}).get('description', '')}".rstrip())
            for f in fs:
                lines.append("  " + declaration(f).replace("\n", "\n  "))
        out_names = [n for n in tb.names() if n not in declared]
        tail = (f"\n// Not declared here, found with tools.search / tools.describe: {', '.join(out_names)}"
                if out_names else "")
        decls = (f"{TOOL_RESULT_TYPE}\n{GLOBAL_DECLARATIONS}\ndeclare const tools: {{\n{BUILTIN_DECLARATIONS}\n"
                 + "\n".join(lines) + f"\n}};{tail}")
        shown = [tb.by_name[n] for n in declared]
        return (_HEAD.replace("{cap}", str(cap_chars)) + "\n\nServer functions (TypeScript; call tools.<name>(args); "
                "each resolves to a ToolResult and never throws):\n```ts\n" + decls + "\n```\nExample:\n```js\n"
                + _example(tb, shown or tb.fns[:2]) + "\n```")

    declared: list[str] = []
    text = build(declared)
    for n in ranked:
        trial = build(declared + [n])
        if tokens_of(json.dumps({"type": "function", "function": {
                "name": TOOL_NAME, "description": trial, "parameters": schema}})) > budget:
            continue
        declared.append(n)
        text = trial
    # keep the registry's own order within the description
    declared = [n for n in tb.names() if n in declared]
    text = build(declared)
    defn = {"type": "function", "function": {"name": TOOL_NAME, "description": text, "parameters": schema}}
    rec = {"budget_tokens": budget, "tokens": tokens_of(json.dumps(defn)), "declared": declared,
           "left_out": [n for n in tb.names() if n not in declared], "query_chars": len(rank_query or "")}
    return defn, rec


def budget_for(covered_defs: list[dict], system_line: str = "") -> int:
    """The inline budget: what the tools the toolbox covers cost today (the served definitions, the proxy's
    estimate), capped at Pi's default 3,000; Pi's figure when nothing is covered (docs/CODEMODE.md)."""
    cost = sum(tokens_of(json.dumps(d)) for d in covered_defs or []) + tokens_of(system_line)
    return min(cost, INLINE_BUDGET_CEILING_TOKENS) if cost else INLINE_BUDGET_CEILING_TOKENS


# ------------------------------------------------------------- the runner --
def available() -> tuple[bool, str]:
    """Can the sandbox run here? (wasmtime importable and the recorded QuickJS WASI module on disk: codemode_sandbox.)
    The offer checks it, so a missing module is a reason in the record, never a broken tool."""
    import codemode_sandbox
    return codemode_sandbox.available()


def error_result(code: str, reason: str, retryable: bool, nxt: str) -> dict:
    """A ToolResult that failed (a function never throws)."""
    return {"ok": False, "text": f"{reason} {nxt}".strip(),
            "error": {"code": code, "reason": reason, "retryable": bool(retryable), "next": nxt}}


def _args_of(tb: Toolbox, name: str, args) -> dict:
    """The arguments object a call passes: an object as is; a bare string for a function whose only required
    parameter is a string (`tools.yama_recall_craft("koota traits")`) is that parameter."""
    if isinstance(args, dict):
        return args
    f = tb.by_name.get(name) or {}
    req = list((f.get("parameters") or {}).get("required") or [])
    props = (f.get("parameters") or {}).get("properties") or {}
    if isinstance(args, str) and len(req) == 1 and (props.get(req[0]) or {}).get("type") == "string":
        return {req[0]: args}
    return {}


def run(script: str, tb: Toolbox, call, *, timeout_s: float, max_calls: int,
        memory_bytes: int = MEMORY_BYTES, concurrency: int = CONCURRENCY, cancelled=None,
        store: dict | None = None) -> dict:
    """Run `script` in the sandbox (QuickJS on WASI under wasmtime, mcp/codemode_sandbox.py). `call(name, args, seq)
    -> ToolResult dict` runs ONE server function in a worker thread (the proxy's bridge; `seq` is the order the
    script ISSUED it in, 1-based); it is called only for names in `tb`, at most `max_calls` times, four at a time.

    Returns the outcome: {ok, reason, error, items, console, wall_ms, calls[{tool, ms, ok, seq}], discovery[...],
    searches, describes, output_chars, dispatched, instantiate_ms}. `reason`: completed | exit | js_error |
    syntax_error | timeout | memory | output_limit | never_settles | crashed | cancelled. Never raises (a script's
    failure is the outcome). The module runs on THIS thread; the calls run on four others."""
    import codemode_sandbox as S
    t0 = time.monotonic()
    out = {"ok": False, "reason": "crashed", "error": None, "items": [], "console": [], "wall_ms": 0, "calls": [],
           "discovery": [], "searches": 0, "describes": 0, "output_chars": 0, "dispatched": 0,
           "instantiate_ms": None, "images": [], "stored": 0, "loaded": 0}
    writes: dict = {}                                            # store() in this run; committed only on success
    committed = store if store is not None else {}
    pool = ThreadPoolExecutor(max_workers=max(int(concurrency), 1), thread_name_prefix="compose-call")
    allowed = set(tb.names())
    st = {"n_calls": 0, "done": None, "kill": None}
    lock = threading.Lock()
    sbx: dict = {}
    timers: list = []                                            # sleep() timers, cancelled when the run ends

    def r_deadline():
        r = sbx.get("run")
        return r.deadline if r is not None else None

    def run_one(seq: int, cid: int, name: str, args) -> None:
        t1 = time.monotonic()
        try:
            res = call(name, _args_of(tb, name, args), seq)
            if not isinstance(res, dict) or "ok" not in res:
                res = error_result("BAD_RESULT", f"{name} returned no result.", False, "call it again.")
        except Exception as e:                                   # noqa: BLE001
            res = error_result("TOOL_RAISED", f"{name} failed ({type(e).__name__}: {str(e)[:160]}).", True,
                               "call it again, or carry on without it.")
        ms = round((time.monotonic() - t1) * 1000)
        with lock:
            out["calls"].append({"tool": name, "ms": ms, "ok": bool(res.get("ok")), "seq": seq})
        r = sbx.get("run")
        if r is not None:
            r.inbox.put({"op": "result", "id": cid, "value": res})

    def on_message(msg: dict) -> list:
        op = msg.get("op")
        if op == "out":
            kind, v = msg.get("kind"), str(msg.get("v") or "")
            out["output_chars"] += len(v)
            if out["output_chars"] > MAX_OUTPUT_CHARS or len(out["items"]) + len(out["console"]) \
                    >= MAX_OUTPUT_CALLS:
                st["kill"] = "output_limit"
                raise S.Killed("output_limit")
            (out["items"] if kind == "text" else out["console"]).append(v)
        elif op == "image":
            if len(out["images"]) < MAX_IMAGES:
                out["images"].append(msg.get("v"))
            out["output_chars"] += 1
        elif op == "q":
            kind, a = msg.get("kind"), msg.get("a") or {}
            if kind == "store":
                k, v = str(a.get("key") or "")[:200], a.get("value")
                if not k or not isinstance(v, str):
                    return [{"op": "r", "v": {"ok": False, "error": "store(key, value) needs a key"}}]
                if len(v) > STORE_VALUE_CHARS:
                    return [{"op": "r", "v": {"ok": False, "error": f"the value is {len(v)} characters; the most "
                                                                    f"one value holds is {STORE_VALUE_CHARS}"}}]
                total = sum(len(x) for kk, x in {**committed, **writes, k: v}.items())
                if total > STORE_TOTAL_CHARS:
                    return [{"op": "r", "v": {"ok": False, "error": f"the store would hold {total} characters; the "
                                                                    f"most it holds is {STORE_TOTAL_CHARS}"}}]
                writes[k] = v
                out["stored"] += 1
                return [{"op": "r", "v": {"ok": True}}]
            if kind == "load":
                k = str(a.get("key") or "")
                out["loaded"] += 1
                return [{"op": "r", "v": writes.get(k, committed.get(k))}]
            if kind == "search":
                v = tb.search(a.get("query"), a.get("opts"))
                out["searches"] += 1
                out["discovery"].append({"kind": "search", "query": str(a.get("query"))[:120],
                                         "hits": [h["name"] for h in v]})
            elif kind == "describe":
                v = tb.describe(a.get("name"))
                out["describes"] += 1
                out["discovery"].append({"kind": "describe", "name": str(a.get("name"))[:80],
                                         "ok": bool(v.get("ok"))})
            elif kind == "namespaces":
                v = tb.namespaces_info(a.get("name"))
            else:
                v = tb.all()
            del out["discovery"][:-40]
            return [{"op": "r", "v": v}]
        elif op == "sleep":
            cid = msg.get("id")
            if isinstance(cid, int):
                left = max(r_deadline() - time.monotonic(), 0) if r_deadline else 0
                t = threading.Timer(min(float(msg.get("ms") or 0) / 1000.0, left),
                                    lambda: sbx["run"].inbox.put({"op": "result", "id": cid, "value": None}))
                t.daemon = True
                timers.append(t)
                t.start()
        elif op == "call":
            cid, name = msg.get("id"), msg.get("name")
            if name not in allowed or not isinstance(cid, int):
                return [{"op": "result", "id": cid, "value": error_result(
                    "NO_SUCH_FUNCTION", f"{str(name)[:80]!r} is not a function of this toolbox.", False,
                    "tools.search(\"what you need\") or tools.all() lists them.")}]
            if st["n_calls"] >= max_calls:
                return [{"op": "result", "id": cid, "value": error_result(
                    "CALL_CAP", f"This request has used its {max_calls} server calls.", False,
                    "carry on with what you have; no more are run in this request.")}]
            st["n_calls"] += 1
            pool.submit(run_one, st["n_calls"], cid, name, msg.get("args"))
        elif op == "done":
            st["done"] = msg
        return []

    try:
        r = S.Run({"op": "init", "script": script, "names": tb.names()}, on_message, memory_bytes=memory_bytes,
                  timeout_s=timeout_s, cancelled=cancelled)
        sbx["run"] = r
        res = r.run()
    except Exception as e:                                       # noqa: BLE001
        res = {"how": "trap", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for t in timers:
            t.cancel()
        out["wall_ms"] = round((time.monotonic() - t0) * 1000)
        out["dispatched"] = st["n_calls"]
    out["instantiate_ms"] = res.get("instantiate_ms")
    done = st["done"]
    if done is not None and done.get("ok") and writes:
        committed.update(writes)                                 # a failed script's store() calls are discarded
    if st["kill"] or res.get("how") == "killed":
        why = st["kill"] or res.get("reason")
        out["reason"] = why
        out["error"] = {"name": "Killed", "message": {
            "timeout": f"the script ran past its {timeout_s:g} s limit",
            "cancelled": "the request was cancelled",
            "output_limit": "the script wrote more than the output limit"}.get(why, str(why))}
    elif done is not None:
        out["ok"] = bool(done.get("ok"))
        out["reason"] = done.get("reason") or ("completed" if out["ok"] else "js_error")
        out["error"] = done.get("error")
        if out["reason"] == "memory":
            out["error"] = {"name": "InternalError", "message": f"out of memory: the script used more than its "
                                                                 f"{memory_bytes // (1024 * 1024)} MiB"}
    elif res.get("reason") == "memory_trap":
        out["reason"] = "memory"
        out["error"] = {"name": "InternalError", "message": f"out of memory: the script used more than its "
                                                             f"{memory_bytes // (1024 * 1024)} MiB"}
    elif res.get("reason") == "stack_overflow":
        out["reason"] = "stack_overflow"
        out["error"] = {"name": "RangeError", "message": "Maximum call stack size exceeded (the sandbox ended the "
                                                         "script)"}
    else:
        out["reason"] = "crashed"
        out["error"] = {"name": "SandboxCrashed",
                        "message": "the sandbox ended without a result (" + str(res.get("reason"))
                        + (f"; {res['stderr'].strip()[-120:]}" if res.get("stderr") else "") + ")"}
    return out


# --------------------------------------------------------------- the result --
_NEXT = {
    "js_error": "fix the line named and call yama_compose_tools again, or call the tools directly.",
    "syntax_error": "fix the syntax and call yama_compose_tools again (the script is the body of an async function).",
    "timeout": "the calls that finished are listed; split the work across scripts or call fewer things in one.",
    "memory": "keep less in memory: slice or filter each result before you collect it.",
    "stack_overflow": "recurse less deeply: loop over the items instead.",
    "output_limit": "print less: return only what you need.",
    "never_settles": "every await must be a tools.* call or a Promise that settles; remove the one that never does.",
    "crashed": "call yama_compose_tools again, or call the tools directly.",
    "cancelled": "",
}


def _where(err: dict | None) -> str:
    """'line N' of the script from an engine stack (the async-function wrapper adds two lines)."""
    m = re.search(r"<input>:(\d+):(\d+)", str((err or {}).get("stack") or ""))
    if not m:
        return ""
    return f" (line {max(int(m.group(1)) - 2, 1)}, column {m.group(2)})"


def format_result(out: dict, cap: int | None = None) -> str:
    """Pi's result: "Script completed" / "Script failed", wall time, the text items (headed `==> text N/M <==`
    when there are several), one <console_output> block, partial output on a failure -- cut to `cap` characters
    (the hidden hop's breaker) with a line that says how to print less."""
    if cap is None:
        cap = OUTPUT_CAP
    secs = out["wall_ms"] / 1000.0
    wall = f"{secs:.1f} s"
    if out["ok"]:
        head = f"Script completed in {wall}"
    else:
        err = out.get("error") or {}
        msg = f"{err.get('name') or 'Error'}: {err.get('message') or out.get('reason')}" \
            if err.get("name") not in (None, "Killed") else str(err.get("message") or out.get("reason"))
        head = f"Script failed in {wall}: {msg}{_where(err)}"
    parts = [head]
    items = out.get("items") or []
    if len(items) == 1:
        parts.append(items[0])
    else:
        parts += [f"==> text {i}/{len(items)} <==\n{t}" for i, t in enumerate(items, 1)]
    if out.get("console"):
        parts.append("<console_output>\n" + "\n".join(out["console"]) + "\n</console_output>")
    bad = [c for c in out.get("calls") or [] if not c.get("ok")]
    if bad:
        names = ", ".join(sorted({c["tool"] for c in bad}))
        parts.append(f"{len(bad)} of {len(out['calls'])} server calls failed ({names}); each result's `error.next` "
                     "says what to do.")
    if not out["ok"]:
        if items or out.get("console"):
            parts.insert(1, "Output before the failure:")
        nxt = _NEXT.get(out.get("reason") or "", "")
        if nxt:
            parts.append("Next: " + nxt)
    text = "\n\n".join(p for p in parts if p != "")
    if len(text) <= cap:
        return text
    note = (f"\n\n[output cut at {cap} of {len(text)} characters: print less -- filter or slice in the script and "
            "return only what you need]")
    kept = text[:max(cap - len(note), 0)]
    nl = kept.rfind("\n")
    if nl > len(kept) // 2:
        kept = kept[:nl]
    return kept.rstrip() + note
