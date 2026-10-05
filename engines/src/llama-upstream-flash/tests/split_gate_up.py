#!/usr/bin/env python3
"""0029: rewrite a GGUF written by test-layer-major --save so that every fused blk.N.ffn_gate_up_exps.weight becomes the
separate blk.N.ffn_gate_exps.weight / blk.N.ffn_up_exps.weight of the real Flash-Next files (a model synthesised from
metadata always fuses them; the real one has three MUL_MAT_IDs per layer, and where the first of them sits in its
scheduler split decides how its weight is copied). The two halves of the fused rows (gate rows, then up rows) are the
new tensors, in the same type; nothing else changes.

    python split_gate_up.py in.gguf out.gguf
"""
import re
import sys

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

SKIP = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}


def main(src: str, dst: str) -> int:
    r = GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(dst, arch)

    for key, f in r.fields.items():
        if key in SKIP or key == "general.architecture":
            continue
        t = f.types
        val = f.contents()
        if t[0] == GGUFValueType.ARRAY and len(val) == 0:
            print("skipping the empty array", key)  # the saver writes one for a per-layer list the model does not use
            continue
        if t[0] == GGUFValueType.ARRAY:
            w.add_key_value(key, val, t[0], sub_type=t[-1])
        else:
            w.add_key_value(key, val, t[0])

    n_split = 0
    for t in r.tensors:
        m = re.fullmatch(r"blk\.(\d+)\.ffn_gate_up_exps\.weight", t.name)
        data = t.data
        if m is None:
            w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
            continue
        il = m.group(1)
        # numpy shape is the reverse of ne: (n_expert, 2*n_ff, row) with row = n_embd (f32) or the row's bytes (quantized)
        n_ff = data.shape[1] // 2
        gate = data[:, :n_ff, :]
        up = data[:, n_ff:, :]
        for name, part in ((f"blk.{il}.ffn_gate_exps.weight", gate), (f"blk.{il}.ffn_up_exps.weight", up)):
            part = part.copy()
            w.add_tensor(name, part, raw_dtype=t.tensor_type)
        n_split += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"split {n_split} fused gate_up tensors; wrote {dst}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
