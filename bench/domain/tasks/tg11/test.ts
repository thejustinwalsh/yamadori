import assert from 'node:assert/strict';
import tgpu, { type TgpuFn, type TgpuSlot } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { clampTonemap, reinhard, shade, shadeClamp, shadeReinhard, tonemapSlot } from './solution.ts';

type Shade = TgpuFn<(...args: [d.Vec3f]) => d.Vec4f>;
const _a: Shade = shade;
const _b: Shade = shadeReinhard;
const _c: Shade = shadeClamp;
type _slot = Expect<Equal<typeof tonemapSlot extends TgpuSlot<any> ? true : false, true>>;
export const _neg = () => {
  // @ts-expect-error -- shade returns a vec4f, not a vec3f
  const _x: TgpuFn<(...args: [d.Vec3f]) => d.Vec3f> = shade;
};

const fnHeader = /fn\s+shade\s*\(\s*(\w+)\s*:\s*vec3f\s*\)\s*->\s*vec4f/;

function check(name: string, fn: unknown, callee: string, other: string) {
  const wgsl = tgpu.resolve([fn as Shade]);
  assert.match(wgsl, fnHeader, `${name}: shade header\n${wgsl}`);
  assert.match(wgsl, new RegExp(`fn\\s+${callee}\\s*\\(`), `${name}: declares ${callee}\n${wgsl}`);
  const body = wgsl.slice(wgsl.search(fnHeader));
  assert.match(body, new RegExp(`\\b${callee}\\s*\\(`), `${name}: shade calls ${callee}\n${wgsl}`);
  assert.doesNotMatch(wgsl, new RegExp(`\\b${other}\\b`), `${name}: must not pull in ${other}\n${wgsl}`);
  assert.match(body, /vec4f\s*\(/, `${name}: returns a vec4f\n${wgsl}`);
}
check('shadeReinhard', shadeReinhard, 'reinhard', 'clampTonemap');
check('shadeClamp', shadeClamp, 'clampTonemap', 'reinhard');
// The slot is what varies: specialising `shade` directly behaves the same.
check('shade.with(clamp)', shade.with(tonemapSlot as TgpuSlot<unknown>, clampTonemap), 'clampTonemap', 'reinhard');

assert.throws(() => tgpu.resolve([shade]), /slot|missing/i, 'resolving shade with the slot unfilled must fail');
void reinhard;
