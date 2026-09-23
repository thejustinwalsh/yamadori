// Shared helpers for `rust` graders with a `wasm_test`. Copied next to each
// task's grader.mjs. Everything fails by throwing; node exits non-zero and
// grade.py records stage `test` with the message.
import { readFileSync } from 'node:fs';

export class GraderFailure extends Error {}

export function assert(cond, msg) {
  if (!cond) throw new GraderFailure(msg);
}

export function eq(actual, expected, msg) {
  const a = typeof actual === 'bigint' || typeof expected === 'bigint'
    ? String(actual) : JSON.stringify(actual);
  const e = typeof actual === 'bigint' || typeof expected === 'bigint'
    ? String(expected) : JSON.stringify(expected);
  if (a !== e) throw new GraderFailure(`${msg}: expected ${e}, got ${a}`);
}

export function close(actual, expected, tol, msg) {
  if (!(Math.abs(actual - expected) <= tol))
    throw new GraderFailure(`${msg}: expected ${expected} +/- ${tol}, got ${actual}`);
}

export function throws(fn, msg) {
  let threw = false;
  try { fn(); } catch { threw = true; }
  if (!threw) throw new GraderFailure(`${msg}: expected a throw`);
}

// ---- a minimal wasm binary reader: types, imports, functions, exports ----

const VALTYPE = { 0x7f: 'i32', 0x7e: 'i64', 0x7d: 'f32', 0x7c: 'f64',
                  0x7b: 'v128', 0x70: 'funcref', 0x6f: 'externref' };

function reader(buf) {
  let pos = 0;
  const r = {
    get pos() { return pos; }, set pos(v) { pos = v; },
    byte() { return buf[pos++]; },
    leb() {           // unsigned LEB128, as a Number (fine below 2^53)
      let result = 0, mul = 1, b;
      do { b = buf[pos++]; result += (b & 0x7f) * mul; mul *= 128; } while (b & 0x80);
      return result;
    },
    name() { const n = r.leb(); const s = new TextDecoder().decode(buf.subarray(pos, pos + n)); pos += n; return s; },
    limits() { const f = r.byte(); r.leb(); if (f & 1) r.leb(); },
  };
  return r;
}

/** {exportName: {kind, params?, results?}} for every export of a module. */
export function wasmSignatures(bytes) {
  const buf = new Uint8Array(bytes);
  const r = reader(buf);
  r.pos = 8;
  const types = [], funcTypes = [], exports = {};
  while (r.pos < buf.length) {
    const id = r.byte(); const size = r.leb(); const end = r.pos + size;
    if (id === 1) {
      const n = r.leb();
      for (let i = 0; i < n; i++) {
        const form = r.byte();
        if (form !== 0x60) throw new Error(`unsupported type form 0x${form.toString(16)}`);
        const params = []; let k = r.leb(); while (k--) params.push(VALTYPE[r.byte()]);
        const results = []; k = r.leb(); while (k--) results.push(VALTYPE[r.byte()]);
        types.push({ params, results });
      }
    } else if (id === 2) {
      const n = r.leb();
      for (let i = 0; i < n; i++) {
        r.name(); r.name(); const kind = r.byte();
        if (kind === 0) funcTypes.push(types[r.leb()]);
        else if (kind === 1) { r.byte(); r.limits(); }
        else if (kind === 2) r.limits();
        else if (kind === 3) { r.byte(); r.byte(); }
        else if (kind === 4) { r.byte(); r.leb(); }
      }
    } else if (id === 3) {
      const n = r.leb();
      for (let i = 0; i < n; i++) funcTypes.push(types[r.leb()]);
    } else if (id === 7) {
      const n = r.leb();
      for (let i = 0; i < n; i++) {
        const name = r.name(); const kind = r.byte(); const idx = r.leb();
        const kinds = ['func', 'table', 'memory', 'global', 'tag'];
        exports[name] = kind === 0 ? { kind: 'func', ...funcTypes[idx] } : { kind: kinds[kind] };
      }
    }
    r.pos = end;
  }
  return exports;
}

/** Assert an export exists with exactly this wasm-level signature. */
export function sig(sigs, name, params, results) {
  const s = sigs[name];
  assert(s, `no export named '${name}' (exports: ${Object.keys(sigs).filter(k => !k.startsWith('__')).join(', ')}) -- missing #[no_mangle] / pub / extern "C"?`);
  assert(s.kind === 'func', `'${name}' is a ${s.kind}, not a function`);
  eq(s.params, params, `wasm params of '${name}'`);
  eq(s.results, results, `wasm results of '${name}'`);
}

/** Instantiate a raw (non-bindgen) module; unexpected imports throw when called. */
export async function load(path) {
  const bytes = readFileSync(path);
  const mod = await WebAssembly.compile(bytes);
  const imports = {};
  for (const imp of WebAssembly.Module.imports(mod)) {
    imports[imp.module] ??= {};
    if (imp.kind === 'function')
      imports[imp.module][imp.name] = () => { throw new GraderFailure(`module called unexpected import ${imp.module}.${imp.name}`); };
  }
  const instance = await WebAssembly.instantiate(mod, imports);
  return { instance, exports: instance.exports, memory: instance.exports.memory,
           sigs: wasmSignatures(bytes) };
}

/** Copy bytes into linear memory via the grader's own allocator. */
export function put(ex, bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes.buffer ?? bytes);
  const ptr = ex.__grader_alloc(u8.length);
  new Uint8Array(ex.memory.buffer, ptr, u8.length).set(u8);
  return ptr;
}

export function view(ex, ptr, len) {
  return new Uint8Array(ex.memory.buffer, ptr, len).slice();
}

export function dv(ex) { return new DataView(ex.memory.buffer); }
