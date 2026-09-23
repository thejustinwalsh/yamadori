import assert from 'node:assert/strict';
import type { TgpuComptime } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { hexToRgb } from './solution.ts';

// Marked for shader-generation-time evaluation (a plain function lacks the marker).
const _ct: TgpuComptime = hexToRgb;
type _ret = Expect<Equal<ReturnType<typeof hexToRgb>, d.v3f>>;
export const _neg = () => {
  // @ts-expect-error -- takes a packed number
  hexToRgb('#ff8000');
};

// The runtime marker typegpu itself checks (core/function/comptime.js isComptimeFn).
const internal = Object.getOwnPropertySymbols(hexToRgb).find((s) => s.description === 'typegpu:0.12.5:$internal');
assert.ok(internal, 'hexToRgb carries typegpu internals');
assert.equal((hexToRgb as unknown as Record<symbol, { isComptime?: boolean }>)[internal]?.isComptime, true,
  'hexToRgb must be created with the comptime helper');

const close = (v: d.v3f, e: [number, number, number], what: string) => {
  assert.equal(v.length, 3, what);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(v[i]! - e[i]) < 1e-6, `${what}[${i}] = ${v[i]}, want ${e[i]}`);
};
close(hexToRgb(0xff8000), [1, 128 / 255, 0], '0xff8000');
close(hexToRgb(0x000000), [0, 0, 0], '0x000000');
close(hexToRgb(0xffffff), [1, 1, 1], '0xffffff');
close(hexToRgb(0x12abef), [0x12 / 255, 0xab / 255, 0xef / 255], '0x12abef');
close(hexToRgb(0x0000ff), [0, 0, 1], '0x0000ff');
