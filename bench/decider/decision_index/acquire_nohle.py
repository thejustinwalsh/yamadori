# Our driver (not part of the harness): fetch every pinned source of the 0.2.1 suite except HLE (gated on the Hub).
from decision_index.suite.build import acquire as acq, adapters_added
from decision_index.suite.build.layout import Layout
L = Layout("work")
nums = [n for n in sorted(acq.NEEDS) if n != 45]
acq.acquire(L, nums, log=lambda s: print(s, flush=True))
adapters_added.acquire(L, list(adapters_added.ORDER), log=lambda s: print(s, flush=True))
print("ACQUIRE DONE", flush=True)
