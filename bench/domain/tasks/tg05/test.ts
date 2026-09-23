import assert from 'node:assert/strict';
import tgpu, { type TgpuVar } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { jitter, seed, tile } from './solution.ts';

type _tile = Expect<Equal<typeof tile, TgpuVar<'workgroup', d.WgslArray<d.F32>>>>;
type _seed = Expect<Equal<typeof seed, TgpuVar<'private', d.U32>>>;
type _jit = Expect<Equal<typeof jitter, TgpuVar<'private', d.Vec2f>>>;
// @ts-expect-error -- the tile is workgroup-scoped, not private
const _p: TgpuVar<'private'> = tile;

const t = tgpu.resolve([tile]);
assert.match(t, /var<\s*workgroup\s*>\s+tile\s*:\s*array<\s*f32\s*,\s*256\s*>\s*;/, t);

const s = tgpu.resolve([seed]);
assert.match(s, /var<\s*private\s*>\s+seed\s*:\s*u32\s*=\s*(?:1u?|u32\(\s*1u?\s*\))\s*;/, s);

const j = tgpu.resolve([jitter]);
const m = /var<\s*private\s*>\s+jitter\s*:\s*vec2f\s*=\s*vec2f\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)\s*;/.exec(j);
assert.ok(m, j);
assert.equal(Number.parseFloat(m[1]), 0.5, j);
assert.equal(Number.parseFloat(m[2]), 1, j);
