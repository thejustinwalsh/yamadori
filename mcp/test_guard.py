#!/usr/bin/env python
"""The indexer's trust boundary, asserted. No real index/repos, no real roots.

WHAT THIS IS GATING

`guard.py` answers two questions whose wrong answers are both silent:

  1. SHOULD THIS FILE BE INDEXED? A credential that reaches the index is fed
     to a model. Named-like-a-secret files are refused by name; the config
     and script extensions are also sniffed for credential-shaped content.
  2. MAY THIS ROOT BE ESTABLISHED? A path that arrived in file content -- a
     tool result, pasted text -- must never be enough to index a new
     directory. Only the harness (`from_trusted=True`) can approve a new
     root, once; after that it stays approved.

NO REAL CONSENT FILE

`YAMADORI_INDEX_DIR` is pointed at a temp directory BEFORE `guard` is imported,
and `YAMADORI_TRUST_ALL_ROOTS` is removed, because both are read at import.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_guard_")
os.environ["YAMADORI_INDEX_DIR"] = os.path.join(_TMP, "repos")
os.environ.pop("YAMADORI_TRUST_ALL_ROOTS", None)

import guard  # noqa: E402

REAL = os.path.abspath(os.path.join(HERE, "..", "index", "repos", "approved_roots.json"))
FILES = os.path.join(_TMP, "files")
os.makedirs(FILES, exist_ok=True)

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def write(name: str, text: str) -> str:
    p = os.path.join(FILES, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def reset_consent() -> None:
    if os.path.exists(guard.CONSENT):
        os.remove(guard.CONSENT)


# Credential-shaped values, assembled so this source file does not itself
# contain one (it would trip the repo's own scanners, and guard's).
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
GH = "ghp_" + "a" * 36
PEM = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIB...\n"
SK = "sk-" + "Z" * 40
JWT = "eyJ" + "a" * 24 + "." + "b" * 24 + ".sig"


# ---------------------------------------------------------------------------


def test_the_fixture_is_not_the_real_consent_file():
    c = os.path.abspath(guard.CONSENT)
    check(c.startswith(os.path.abspath(_TMP)), "the consent file is in a temp dir", c)
    check(c != REAL, "and is not index/repos/approved_roots.json")


def test_credential_names_are_refused():
    for name in ("secrets.json", "Secrets.JSON", ".env", ".env.production",
                 ".npmrc", ".netrc", ".pgpass", "id_rsa", "id_ed25519",
                 "server.pem", "cert.pfx", "store.p12", "tls.key", "app.keystore",
                 "release.jks", "credentials.toml", "terraform.tfstate",
                 ".htpasswd", "service_account-prod.json", "my_api_key.txt",
                 "password.txt"):
        ok, why = guard.should_index(os.path.join("repo", "sub", name))
        check(not ok and "credential" in why, f"{name} is refused by name", why)


def test_ordinary_source_is_allowed():
    for name in ("main.ts", "lib.rs", "README.md", "package.json", "Cargo.toml",
                 "config.yaml", "build.ps1", "env.ts", "keyboard.ts",
                 "identity.rs"):
        p = write(name, "export const x = 1;\n")
        ok, why = guard.should_index(p)
        check(ok and why == "", f"{name} with ordinary content is indexed", why)
    ok, _ = guard.should_index(os.path.join(FILES, "does-not-exist.json"))
    check(ok, "an unreadable file of a sniffed type is not refused for content")


def test_credential_content_is_refused_in_sniffed_types():
    # Neutral file names: "token" or "key" in the NAME would be refused by
    # name first, and this test would pass without sniffing anything.
    for n, (label, value) in enumerate((("AWS key id", AWS), ("GitHub token", GH),
                                        ("private key", PEM), ("sk- key", SK),
                                        ("JWT", JWT))):
        p = write(f"settings_{n}.yaml", f"x: 1\nval: {value}\n")
        ok, why = guard.should_index(p)
        check(not ok and "credential-shaped" in why,
              f"a {label} inside a .yaml is refused", why)
    p = write("notes.md", "use AKIA as the prefix; no real key here\n")
    ok, _ = guard.should_index(p)
    check(ok, "a bare prefix without the full shape is not a false positive")


def test_content_is_sniffed_in_every_file_and_names_only_in_data_files():
    """A real key format in source is a leak like any other, so every file's
    content is checked. Credential WORDS in a source file's NAME are ordinary
    code, so they only block data/config files."""
    p = write("fixture.ts", f"const example = '{GH}';\n")
    ok, why = guard.should_index(p)
    check(not ok and "credential-shaped" in why,
          "a .ts file carrying a real token format is refused", why)
    for name in ("tokenizer.ts", "password-reset.tsx", "useApiKey.ts",
                 "secretsManager.py", "credential_store.rs"):
        p = write(name, "export const x = 1;\n")
        ok, why = guard.should_index(p)
        check(ok, f"source named {name} is indexed", why)
    for name in ("secrets.json", "credentials.toml", "api_key.txt",
                 "tokens.yaml", ".env.local", "server.pem"):
        check(guard.is_secret_path(name),
              f"data/credential file {name} is still refused by name")


def test_a_secret_past_the_sniff_window_is_not_seen():
    """Pinned so a change to MAX_SNIFF is a decision, not an accident."""
    p = write("big.json", " " * (guard.MAX_SNIFF + 10) + AWS)
    ok, _ = guard.should_index(p)
    check(ok, f"only the first {guard.MAX_SNIFF} characters are sniffed")


def test_a_root_from_file_content_cannot_be_established():
    reset_consent()
    root = os.path.join(_TMP, "someone-elses-private-dir")
    ok, why = guard.may_establish(root, from_trusted=False)
    check(not ok, "an untrusted new root is refused")
    check("operator" in why, "and the reason says only the operator approves", why)
    check(not guard.is_approved(root), "and the refusal approves nothing")
    check(guard.approved_roots() == [], "the consent file is still empty")


def test_a_request_can_never_approve_a_root():
    """The hole closed on 2026-09-22: a path in the system or user message --
    both written by any key holder -- used to be approved on first sight and
    indexed. Neither a 'trusted' mention nor anything else in a request may
    approve a directory on this server."""
    reset_consent()
    root = os.path.join(_TMP, "my-repo")
    os.makedirs(root, exist_ok=True)
    ok, why = guard.may_establish(root, from_trusted=True)
    check(not ok, "a harness-declared (system/user message) root is refused")
    check("operator" in why, "and the reason says only the operator approves", why)
    check(guard.approved_roots() == [], "nothing was written to the consent file")
    import proxy
    got = proxy.resolve_repo([
        {"role": "system", "content": f"Current working directory: {root}"},
        {"role": "user", "content": f"cd {root} && look at {root}\src"}])
    check(got[0] is None, "proxy.resolve_repo returns no root for a real "
                          "directory named in the messages", str(got))


def test_an_operator_approved_root_is_remembered():
    reset_consent()
    root = os.path.join(_TMP, "my-repo")
    guard.approve(root, how="cli")
    check(guard.is_approved(root), "an operator-approved root is approved")
    with open(guard.CONSENT, encoding="utf-8") as f:
        d = json.load(f)
    entry = d.get(os.path.abspath(root), {})
    check(entry.get("how") == "cli",
          "the consent record says how it was approved", json.dumps(entry))
    ok, _ = guard.may_establish(root, from_trusted=False)
    check(ok, "once approved, later untrusted mentions are fine")
    check(guard.is_approved(root.upper()) and guard.is_approved(root + os.sep + "."),
          "approval matching ignores case and normalises the path")
    check(not guard.is_approved(os.path.join(root, "sub")),
          "a subdirectory is not implicitly approved")
    check(not guard.is_approved(os.path.dirname(root)),
          "nor is the parent")


def test_revoke_removes_exactly_one_root():
    reset_consent()
    a, b = os.path.join(_TMP, "repo-a"), os.path.join(_TMP, "repo-b")
    guard.approve(a)
    guard.approve(b)
    check(guard.revoke(a.upper()), "revoke matches case-insensitively")
    check(not guard.is_approved(a) and guard.is_approved(b),
          "and removes only that root")
    check(guard.revoke(a) is False, "revoking again reports it was not approved")


def test_a_corrupt_consent_file_fails_closed():
    reset_consent()
    root = os.path.join(_TMP, "was-approved")
    guard.approve(root)
    with open(guard.CONSENT, "w", encoding="utf-8") as f:
        f.write("{ corrupt")
    check(not guard.is_approved(root),
          "a corrupt consent file approves nothing (fails closed)")
    ok, _ = guard.may_establish(root, from_trusted=False)
    check(not ok, "so an untrusted mention cannot establish it")


def test_there_is_no_open_mode():
    check(not hasattr(guard, "OPEN"), "no switch approves every root")
    old = os.environ.get("YAMADORI_TRUST_ALL_ROOTS")
    os.environ["YAMADORI_TRUST_ALL_ROOTS"] = "1"
    try:
        check(not guard.is_approved(os.path.join(_TMP, "anything")),
              "and the old variable approves nothing")
    finally:
        if old is None:
            os.environ.pop("YAMADORI_TRUST_ALL_ROOTS", None)
        else:
            os.environ["YAMADORI_TRUST_ALL_ROOTS"] = old


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_consent_file,
               test_credential_names_are_refused,
               test_ordinary_source_is_allowed,
               test_credential_content_is_refused_in_sniffed_types,
               test_content_is_sniffed_in_every_file_and_names_only_in_data_files,
               test_a_secret_past_the_sniff_window_is_not_seen,
               test_a_root_from_file_content_cannot_be_established,
               test_a_request_can_never_approve_a_root,
               test_an_operator_approved_root_is_remembered,
               test_revoke_removes_exactly_one_root,
               test_a_corrupt_consent_file_fails_closed,
               test_there_is_no_open_mode):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    print(f"  temp consent file: {os.path.abspath(guard.CONSENT)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
