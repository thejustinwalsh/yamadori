#!/usr/bin/env python
"""Alias: the egress gate moved to bench/sandbox/egress_gate.py on
2026-09-26 (SELF-IMPROVEMENT-LOG #48/#49). `import egress_gate` here IS that
module. The gate container runs the moved file (sandbox_net.gate_argv
mounts bench/sandbox at /sandbox)."""
import importlib.util
import os
import sys

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "sandbox", "egress_gate.py")
_KEY = "_bench_sandbox_egress_gate"

if _KEY not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_KEY, _PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_KEY] = _mod
    _spec.loader.exec_module(_mod)

if __name__ == "__main__":
    sys.exit(sys.modules[_KEY].main())
sys.modules[__name__] = sys.modules[_KEY]
