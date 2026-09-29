#!/usr/bin/env python
"""Alias: the sandbox network moved to bench/sandbox/sandbox_net.py on
2026-09-26, when SWE-bench and harness_box.py started using it too
(SELF-IMPROVEMENT-LOG #48/#49). `import sandbox_net` here IS that module --
the same object, so a test that patches `sandbox_net._docker` patches the
real one.

    python bench/octopus/sandbox_net.py verify|plan     # as before
"""
import importlib.util
import os
import sys

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "sandbox", "sandbox_net.py")
_KEY = "_bench_sandbox_sandbox_net"

if _KEY not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_KEY, _PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_KEY] = _mod
    _spec.loader.exec_module(_mod)

if __name__ == "__main__":
    sys.exit(sys.modules[_KEY].main())
sys.modules[__name__] = sys.modules[_KEY]
