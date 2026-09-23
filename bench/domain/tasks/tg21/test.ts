import assert from 'node:assert/strict';
import tgpu, { type TgpuFragmentFn, type TgpuVertexFn } from 'typegpu';
import type { Equal, Expect, IsAny } from './assert_types.ts';
import { mainFragment, mainVertex } from './solution.ts';

const _v: TgpuVertexFn = mainVertex as TgpuVertexFn<any, any>;
const _f: TgpuFragmentFn = mainFragment as TgpuFragmentFn<any, any>;
type _na = Expect<Equal<IsAny<typeof mainVertex> | IsAny<typeof mainFragment>, false>>;
export const _neg = () => {
  // @ts-expect-error -- a vertex entry point is not a fragment entry point
  const _x: TgpuFragmentFn = mainVertex;
};

type Field = { type: string; location?: number; builtin?: string; interpolate: string };

/** Every `@attr(...)... name: type` declaration (struct members and parameters). */
function fields(wgsl: string): Map<string, Field> {
  const out = new Map<string, Field>();
  const re = /((?:@\w+(?:\([^()]*\))?\s*)+)(\w+)\s*:\s*([\w<>]+)/g;
  for (const m of wgsl.matchAll(re)) {
    const attrs = m[1]!;
    const loc = /@location\(\s*(\d+)\s*\)/.exec(attrs);
    const bi = /@builtin\(\s*(\w+)\s*\)/.exec(attrs);
    const interp = /@interpolate\(\s*([^)]*?)\s*\)/.exec(attrs);
    // Default interpolation is perspective/center; normalise the spellings.
    const kind = (interp?.[1] ?? 'perspective').split(',').map((s) => s.trim());
    const norm = kind[0] === 'flat' ? 'flat' : `${kind[0]}${kind[1] && kind[1] !== 'center' ? ',' + kind[1] : ''}`;
    out.set(m[2]!, { type: m[3]!, location: loc ? Number(loc[1]) : undefined, builtin: bi?.[1], interpolate: norm });
  }
  return out;
}

const vs = tgpu.resolve([mainVertex]);
const fs = tgpu.resolve([mainFragment]);
assert.match(vs, /@vertex\s+fn\s+mainVertex\s*\(/, vs);
assert.match(fs, /@fragment\s+fn\s+mainFragment\s*\(/, fs);
assert.equal((vs.match(/\bfn\s+mainVertex\b/g) ?? []).length, 1, 'one vertex header\n' + vs);

const V = fields(vs);
const F = fields(fs);
const need = (m: Map<string, Field>, k: string, src: string) => {
  const f = m.get(k);
  assert.ok(f, `missing ${k}\n${src}`);
  return f;
};
assert.deepEqual([need(V, 'position', vs).location, need(V, 'position', vs).type], [0, 'vec2f'], 'position @location(0) vec2f\n' + vs);
assert.deepEqual([need(V, 'cellId', vs).location, need(V, 'cellId', vs).type], [1, 'u32'], 'cellId @location(1) u32\n' + vs);
assert.equal(need(V, 'pos', vs).builtin, 'position', 'pos is @builtin(position)\n' + vs);

const vuv = need(V, 'uv', vs), vcell = need(V, 'cell', vs);
const fuv = need(F, 'uv', fs), fcell = need(F, 'cell', fs);
assert.equal(vuv.type, 'vec2f', vs);
assert.equal(vcell.type, 'u32', vs);
assert.equal(vcell.interpolate, 'flat', 'integer vertex output needs @interpolate(flat)\n' + vs);
assert.equal(fcell.interpolate, 'flat', 'integer fragment input needs @interpolate(flat)\n' + fs);
assert.notEqual(vuv.interpolate, 'flat', 'uv is perspective-interpolated\n' + vs);
assert.ok(vuv.location !== undefined && vcell.location !== undefined && vuv.location !== vcell.location, vs);
assert.equal(fuv.location, vuv.location, `uv location must link: vertex ${vuv.location}, fragment ${fuv.location}\n${vs}\n${fs}`);
assert.equal(fcell.location, vcell.location, `cell location must link: vertex ${vcell.location}, fragment ${fcell.location}\n${vs}\n${fs}`);
assert.equal(fuv.interpolate, vuv.interpolate, 'uv interpolation must match across stages');

assert.match(fs, /->\s*@location\(\s*0\s*\)\s*vec4f/, 'fragment returns a vec4f at location 0\n' + fs);
assert.match(fs, /%\s*2u/, fs);
assert.match(vs, /\*\s*0\.5\s*\+\s*0\.5/, vs);
