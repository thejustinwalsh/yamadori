import assert from 'node:assert/strict';
import tgpu, { type TgpuAccessor, type TgpuFn } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { resolveTinted, tintAccess, tinted } from './solution.ts';

type _acc = Expect<Equal<typeof tintAccess, TgpuAccessor<d.Vec3f>>>;
const _fn: TgpuFn<(...args: [d.Vec3f]) => d.Vec3f> = tinted;
type _r = Expect<Equal<Parameters<typeof resolveTinted>, [color: d.v3f]>>;
export const _neg = () => {
  // @ts-expect-error -- the accessor is typed vec3f; an f32 accessor is a different type
  const _x: TgpuAccessor<d.F32> = tintAccess;
};

const header = /fn\s+tinted\s*\(\s*(\w+)\s*:\s*vec3f\s*\)\s*->\s*vec3f\s*\{([\s\S]*)\}/;

/** The components of every vec3f(...) literal in `code`, splats expanded. */
function vec3Literals(code: string): number[][] {
  return [...code.matchAll(/vec3f\s*\(([^()]*)\)/g)].map((m) => {
    const parts = m[1]!.split(',').map((s) => Number.parseFloat(s.trim()));
    return parts.length === 1 ? [parts[0]!, parts[0]!, parts[0]!] : parts;
  });
}
const hasVec = (code: string, v: number[]) =>
  vec3Literals(code).some((l) => l.length === 3 && l.every((x, i) => Math.abs(x - v[i]!) < 1e-6));

// Default value.
const def = tgpu.resolve([tinted]);
assert.match(def, header, def);
assert.ok(hasVec(def, [1, 1, 1]), 'default tint vec3f(1,1,1) inlined\n' + def);

// A literal, through the helper.
const lit = resolveTinted(d.vec3f(1, 0.5, 0.25));
assert.match(lit, header, lit);
assert.ok(hasVec(lit, [1, 0.5, 0.25]), 'bound literal inlined\n' + lit);
assert.ok(!hasVec(lit, [1, 1, 1]), 'the default is not used once a value is bound\n' + lit);

// A module constant: referenced by name.
const GOLD = tgpu.const(d.vec3f, d.vec3f(1, 0.8, 0)).$name('GOLD');
const withConst = tgpu.resolve([tinted.with(tintAccess, GOLD)]);
assert.match(withConst, /const\s+GOLD\s*:\s*vec3f/, withConst);
assert.match(header.exec(withConst)?.[2] ?? '', /\bGOLD\b/, 'tinted reads GOLD\n' + withConst);

// A function returning vec3f: must be called.
const brand = tgpu.fn([], d.vec3f)`() -> vec3f { return vec3f(0.2, 0.4, 0.8); }`.$name('brandColor');
const withFn = tgpu.resolve([tinted.with(tintAccess, brand)]);
assert.match(withFn, /fn\s+brandColor\s*\(\s*\)/, withFn);
assert.match(header.exec(withFn)?.[2] ?? '', /\bbrandColor\s*\(\s*\)/, 'tinted calls brandColor()\n' + withFn);
