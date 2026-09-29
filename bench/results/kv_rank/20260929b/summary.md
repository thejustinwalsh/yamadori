line N = 141824 (margin 1000 MiB, -c 349184, host RAM pinned ~7.22 GB)
  round N=147456 -c 360448: min free 805 MiB 
  round N=141824 -c 349184: min free 1001 MiB 

| stage | numbers | max span by seq | moves (ms) |
|---|---|---|---|
| cap_8192 | {"depth": 8192, "decode_median": 69.06} | {'0': 8704, '1': 0, '2': 0} | 0 (0) |
| cap_65536 | {"depth": 65536, "decode_median": 59.78} | {'0': 65792, '1': 0, '2': 0} | 0 (0) |
| cap_140544 | {"depth": 140544, "decode_median": 55.74} | {'0': 140800, '1': 0, '2': 0} | 0 (0) |
| concurrent_spill | {"A_depth": 133632, "B_depth": 32768, "A_alone_B_idle_median": 57.1, "A_concurrent_median": 17.86, "B_concurrent_median": 18.68} | {'0': 134400, '1': 167168, '2': 0} | 3 (8.4) |
| concurrent_below | {"A_concurrent_median": 34.02, "B_concurrent_median": 34.04} | {'0': 66048, '1': 66048, '2': 0} | 0 (0) |
| concurrent_second_arrived_first | {"A_prefill": {"prompt_n": 100868, "prompt_ms": 231055.818, "prompt_per_second": 436.55252169413023, "predicted_n": 256, "predicted_per_second": 56.285227930893214, "draft_n": 191, "draft_n_accepted": 158, "wall_s": 243. | {'0': 140800, '1': 33024, '2': 0} | 26 (621.8) |
| child_baseline_no_child | {"A_continue": {"prompt_n": 257, "prompt_ms": 854.26, "prompt_per_second": 300.84517594175077, "predicted_n": 256, "predicted_per_second": 59.40667409521888, "draft_n": 184, "draft_n_accepted": 163, "wall_s": 5.22}} | {'0': 138496, '1': 0, '2': 0} | 0 (0) |
| child_49152 | {"size": 49152, "child": {"prompt_n": 49152, "prompt_ms": 69097.981, "prompt_per_second": 711.3377162206809, "predicted_n": 256, "predicted_per_second": 58.247051557092156, "draft_n": 225, "draft_n_accepted": 142, "wall_ | {'0': 138496, '1': 0, '2': 140544} | 47 (1129.4) |
| child_65536 | {"size": 65536, "child": {"prompt_n": 16388, "prompt_ms": 28096.179, "prompt_per_second": 583.2821608945472, "predicted_n": 256, "predicted_per_second": 62.101636268524565, "draft_n": 203, "draft_n_accepted": 153, "wall_ | {'0': 142080, '1': 0, '2': 65792} | 20 (920.7) |
| child_98304 | {"size": 98304, "child": {"prompt_n": 32772, "prompt_ms": 65622.858, "prompt_per_second": 499.39915753135904, "predicted_n": 256, "predicted_per_second": 58.672847793659336, "draft_n": 199, "draft_n_accepted": 155, "wall | {'0': 138496, '1': 0, '2': 98560} | 35 (1599.0) |

pass  alone at 8192: the conversation's span stays at or below the line, nothing moves  {"span": 8704, "line": 141824, "moves": 0}
pass  alone at 65536: the conversation's span stays at or below the line, nothing moves  {"span": 65792, "line": 141824, "moves": 0}
pass  alone at 140544: the conversation's span stays at or below the line, nothing moves  {"span": 140800, "line": 141824, "moves": 0}
pass  a second conversation beside the primary: the primary's span stays at or below the line  {"spans": {"0": 134400, "1": 167168, "2": 0}, "line": 141824}
pass  the second conversation arrived first: the primary still ends at or below the line (the engine moved the lower rank out)  {"spans": {"0": 140800, "1": 33024, "2": 0}, "line": 141824, "moves": 26, "move_ms": 621.8}
pass  child 49152: the child ran at or below the line, and so did its main after it  {"spans": {"0": 138496, "1": 0, "2": 140544}, "line": 141824, "moves": 47, "move_ms": 1129.4}
FAIL  child 65536: the child ran at or below the line, and so did its main after it  {"spans": {"0": 142080, "1": 0, "2": 65792}, "line": 141824, "moves": 20, "move_ms": 920.7}
pass  child 98304: the child ran at or below the line, and so did its main after it  {"spans": {"0": 138496, "1": 0, "2": 98560}, "line": 141824, "moves": 35, "move_ms": 1599.0}
