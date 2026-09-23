#!/usr/bin/env python
"""Accounts and API-key auth, asserted. No network, no real registry.

WHAT THIS IS GATING

`accounts.identify()` is the only thing between a remote caller and another
account's indexes. The promises it makes, each asserted below:

  1. No registry means single-user and open -- and ONLY an absent registry
     means that. A registry that exists but cannot be read fails CLOSED.
     (Until 2026-09-22 it failed open: a corrupt accounts.json came back as
     {} and every caller was admitted as `anonymous`.)
  2. Keys are stored hashed. The plaintext key appears nowhere on disk.
  3. A wrong, missing or non-Bearer key is refused, with a reason.
  4. One account's directory cannot be named by another, and an account id
     cannot walk out of the store.

NO REAL REGISTRY

`YAMADORI_ACCOUNTS_DIR` is pointed at a temp directory BEFORE `accounts` is
imported, because the module reads it at import time. The real
index/accounts/accounts.json is never opened, read or printed -- the first
test asserts the paths differ, and no test output ever contains a key.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_accounts_")
os.environ["YAMADORI_ACCOUNTS_DIR"] = os.path.join(_TMP, "accounts")

import accounts  # noqa: E402

REAL_STORE = os.path.abspath(os.path.join(HERE, "..", "index", "accounts"))

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def reset() -> None:
    """An empty store: no registry file at all."""
    for p in (accounts.REGISTRY, accounts.REGISTRY + ".tmp"):
        if os.path.exists(p):
            os.remove(p)


def raw_registry() -> str:
    with open(accounts.REGISTRY, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------


def test_the_fixture_is_not_the_real_registry():
    store = os.path.abspath(accounts.STORE)
    check(store.startswith(os.path.abspath(_TMP)),
          "the account store is a temp directory", store)
    check(store != REAL_STORE, "and is not index/accounts")
    check(os.path.abspath(accounts.REGISTRY).startswith(os.path.abspath(_TMP)),
          "and the registry file is inside it")


def test_no_registry_is_single_user_and_open():
    reset()
    check(not accounts.multi_tenant(), "no registry file means single-user")
    who, why = accounts.identify(None)
    check(who == accounts.ANON, "a request with no key is anonymous", str(who))
    check(bool(why), "and says why", why)
    who, _ = accounts.identify("Bearer anything-at-all")
    check(who == accounts.ANON,
          "any key is accepted while no account exists", str(who))


def test_a_key_is_stored_only_as_its_hash():
    reset()
    key = accounts.create("alice")
    check(key.startswith("ym-") and len(key) > 20,
          "create() mints a long ym- key", f"length {len(key)}")
    text = raw_registry()
    check(key not in text, "the plaintext key is nowhere in the registry")
    check(key[3:] not in text, "not even without its prefix")
    d = json.loads(text)
    want = hashlib.sha256(("yamadori/" + key).encode()).hexdigest()
    check(list(d) == [want], "the registry is keyed by the salted SHA-256",
          f"{len(d)} entries")
    meta = d.get(want, {})
    check(meta.get("label") == "alice" and meta.get("uses") == 0,
          "with the label and a zero use count", json.dumps(meta)[:120])
    check(not os.path.exists(accounts.REGISTRY + ".tmp"),
          "the atomic-write temp file does not linger")


def test_one_account_switches_the_server_to_keys_required():
    reset()
    key = accounts.create("alice")
    check(accounts.multi_tenant(), "one account makes the server multi-tenant")

    who, why = accounts.identify(None)
    check(who is None, "no Authorization header is refused", str(who))
    check("no api key" in why.lower(), "and the reason names the missing key", why)

    who, why = accounts.identify("")
    check(who is None, "an empty header is refused", str(who))

    who, why = accounts.identify("Basic " + key)
    check(who is None, "a non-Bearer scheme is refused even with a good key",
          str(who))

    who, why = accounts.identify("Bearer ym-not-a-real-key")
    check(who is None, "an unknown key is refused", str(who))
    check("unrecognised" in why, "and the reason says the key is unknown", why)

    who, why = accounts.identify("Bearer " + key[:-1])
    check(who is None, "a key one character short is refused", str(who))


def test_the_right_key_identifies_its_account():
    reset()
    key = accounts.create("alice")
    h = hashlib.sha256(("yamadori/" + key).encode()).hexdigest()
    who, why = accounts.identify("Bearer " + key)
    check(who == h[:16], "the right key maps to its account id",
          f"got {who!r}")
    check("alice" in why, "and the reason names the account's label", why)

    who2, _ = accounts.identify("bearer   " + key + "  ")
    check(who2 == who,
          "the scheme is case-insensitive and surrounding space is ignored",
          f"got {who2!r}")

    meta = json.loads(raw_registry())[h]
    check(meta.get("uses") == 2, "each identification is counted",
          str(meta.get("uses")))
    check(isinstance(meta.get("last_seen"), float), "and last_seen is recorded")


def test_two_accounts_are_told_apart():
    reset()
    ka = accounts.create("alice")
    kb = accounts.create("bob")
    check(ka != kb, "two keys are never the same")
    a, ra = accounts.identify("Bearer " + ka)
    b, rb = accounts.identify("Bearer " + kb)
    check(a is not None and b is not None and a != b,
          "each key identifies a different account", f"{a!r} vs {b!r}")
    check("alice" in ra and "bob" in rb, "each with its own label", f"{ra} / {rb}")
    check(accounts.account_dir(a) != accounts.account_dir(b),
          "and each account has its own directory")


def test_an_unreadable_registry_fails_closed():
    """THE BUG THIS FILE FOUND. A corrupt registry used to read as {} --
    single-user mode -- so every caller was admitted as `anonymous`."""
    reset()
    good = accounts.create("alice")
    with open(accounts.REGISTRY, "w", encoding="utf-8") as f:
        f.write('{"truncated": ')
    check(accounts.multi_tenant(),
          "a corrupt registry still counts as multi-tenant")
    who, why = accounts.identify(None)
    check(who is None, "a request with no key is refused, not admitted",
          str(who))
    who, why = accounts.identify("Bearer " + good)
    check(who is None,
          "even a once-valid key is refused while nothing can be checked",
          str(who))
    check("unreadable" in why and "refused" in why,
          "the reason says the registry is unreadable and the request refused",
          why)
    check(accounts.ANON not in (who or ""), "nobody is ever anonymous here")

    try:
        accounts.create("mallory")
        overwrote = True
    except accounts.RegistryUnreadable:
        overwrote = False
    check(not overwrote,
          "create() refuses instead of overwriting a corrupt registry")
    check(raw_registry() == '{"truncated": ',
          "and the corrupt file is left for the operator to recover")


def test_a_registry_of_the_wrong_shape_fails_closed():
    reset()
    with open(accounts.REGISTRY, "w", encoding="utf-8") as f:
        json.dump(["not", "an", "object"], f)
    who, _ = accounts.identify(None)
    check(who is None, "a JSON list is not a registry: refused", str(who))


def test_an_empty_registry_object_is_single_user():
    """`{}` is a registry with no accounts, which is exactly single-user."""
    reset()
    with open(accounts.REGISTRY, "w", encoding="utf-8") as f:
        f.write("{}")
    who, _ = accounts.identify(None)
    check(who == accounts.ANON, "an empty registry object is open", str(who))


def test_account_dir_cannot_leave_the_store():
    reset()
    store = os.path.abspath(accounts.STORE)
    for hostile in ("../../etc", "..\\..\\Windows", "/abs/path", "a/../../b",
                    "C:\\Users", "x" * 200):
        p = os.path.abspath(accounts.account_dir(hostile))
        check(os.path.dirname(p) == store,
              f"account_dir({hostile[:20]!r}) stays one level under the store",
              p)
        check(len(os.path.basename(p)) <= 32, "and its name is at most 32 chars")
    check(os.path.basename(accounts.account_dir("")) == accounts.ANON,
          "an empty id falls back to the anonymous directory")
    check(os.path.basename(accounts.account_dir("../..")) == accounts.ANON,
          "so does an id made only of path punctuation")


def test_watchlist_stores_the_remote_once_and_per_account():
    reset()
    check(accounts.watchlist("acct1") == [], "a new account watches nothing")
    accounts.watch("acct1", "https://example.invalid/r.git", note="main repo")
    accounts.watch("acct1", "https://example.invalid/r.git", note="again")
    rows = accounts.watchlist("acct1")
    check(len(rows) == 1, "watching the same remote twice keeps one row",
          str(len(rows)))
    check(rows and rows[0]["note"] == "main repo", "and the first note stands")
    check(set(rows[0]) == {"remote", "note", "added"} if rows else False,
          "a row holds the remote, a note and a time -- no credential field",
          str(sorted(rows[0]) if rows else None))
    check(accounts.watchlist("acct2") == [],
          "another account does not see it")

    with open(os.path.join(accounts.account_dir("acct3"), "repos.json"),
              "w", encoding="utf-8") as f:
        f.write("not json")
    check(accounts.watchlist("acct3") == [],
          "a corrupt watchlist reads as empty rather than raising")


def test_usage_counts_real_bytes():
    reset()
    d = accounts.account_dir("sizer")
    with open(os.path.join(d, "blob.bin"), "wb") as f:
        f.write(b"x" * 5000)
    os.makedirs(os.path.join(d, "sub"), exist_ok=True)
    with open(os.path.join(d, "sub", "more.bin"), "wb") as f:
        f.write(b"y" * 1000)
    accounts.watch("sizer", "https://example.invalid/s.git")
    u = accounts.usage("sizer")
    repos_bytes = os.path.getsize(os.path.join(d, "repos.json"))
    check(u["bytes"] == 6000 + repos_bytes,
          "usage sums every file, including subdirectories",
          f"{u['bytes']} vs {6000 + repos_bytes}")
    check(u["repos"] == 1 and u["account"] == "sizer",
          "and reports the watched repo count", json.dumps(u))


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_registry,
               test_no_registry_is_single_user_and_open,
               test_a_key_is_stored_only_as_its_hash,
               test_one_account_switches_the_server_to_keys_required,
               test_the_right_key_identifies_its_account,
               test_two_accounts_are_told_apart,
               test_an_unreadable_registry_fails_closed,
               test_a_registry_of_the_wrong_shape_fails_closed,
               test_an_empty_registry_object_is_single_user,
               test_account_dir_cannot_leave_the_store,
               test_watchlist_stores_the_remote_once_and_per_account,
               test_usage_counts_real_bytes):
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
    print(f"  temp store: {os.path.abspath(accounts.STORE)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
