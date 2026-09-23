import assert from 'node:assert/strict';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { LightUniform, lightUniformSize } from './solution.ts';

// The JS-side value has exactly the four fields: no padding members.
type _v = Expect<Equal<d.Infer<typeof LightUniform>,
  { color: d.v3f; intensity: number; direction: d.v3f; castsShadow: number }>>;
// @ts-expect-error -- a padding field would be a fifth key
const _noPad: d.Infer<typeof LightUniform> = { color: d.vec3f(), intensity: 0, direction: d.vec3f(), castsShadow: 0, _pad: 0 };

const off = (f: (l: any) => unknown) => d.memoryLayoutOf(LightUniform, f).offset;
assert.equal(off((l) => l.intensity), 12, 'intensity packs right after color');
assert.equal(off((l) => l.direction), 32, 'intensity must occupy 16 bytes (12..28), direction aligned to 32');
assert.equal(off((l) => l.castsShadow), 48, 'castsShadow must be aligned to 16 (48)');
assert.equal(d.sizeOf(LightUniform), 64, 'struct size');
assert.equal(lightUniformSize, 64, 'lightUniformSize');

const wgsl = tgpu.resolve([LightUniform]);
// The run of attributes written in front of a member, e.g. "@align(16) @size(4) ".
const attrsOf = (field: string) =>
  new RegExp('((?:@\\w+\\(\\s*\\d+\\s*\\)\\s*)*)' + field + '\\s*:').exec(wgsl)?.[1] ?? '';
assert.match(attrsOf('intensity'), /@size\(\s*16\s*\)/, 'intensity carries @size(16)\n' + wgsl);
assert.match(attrsOf('castsShadow'), /@align\(\s*16\s*\)/, 'castsShadow carries @align(16)\n' + wgsl);
assert.match(wgsl, /color\s*:\s*vec3f[\s\S]*intensity[\s\S]*direction\s*:\s*vec3f[\s\S]*castsShadow/, 'field order');
