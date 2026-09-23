import { load, sig, eq, put } from './wasm_helpers.mjs';

const { exports: ex, sigs } = await load(process.argv[2]);
// usize and pointers are i32 on wasm32; i64 stays i64 across the C ABI.
sig(sigs, 'sum_i32', ['i32', 'i32'], ['i64']);

const arr = new Int32Array([1, -2, 3, 2147483647, 2147483647]);
const ptr = put(ex, new Uint8Array(arr.buffer));
eq(ex.sum_i32(ptr, arr.length), 4294967296n, 'sum with i32 overflow');
eq(ex.sum_i32(ptr, 3), 2n, 'prefix sum');
eq(ex.sum_i32(0, 5), 0n, 'null pointer');
eq(ex.sum_i32(ptr, 0), 0n, 'zero length');
// the caller still owns the array: it is intact and freeable
eq(Array.from(new Int32Array(ex.memory.buffer, ptr, 5)), Array.from(arr), 'input untouched');
ex.__grader_free(ptr, arr.byteLength);
