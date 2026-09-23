import assert from 'node:assert/strict';
import tgpu from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { Camera, Particle, sceneLayout } from './solution.ts';

type L = typeof sceneLayout;
type _cam = Expect<Equal<L['$']['camera'], d.InferGPU<typeof Camera>>>;
type _par = Expect<Equal<L['$']['particles'], d.InferGPU<typeof Particle>[]>>;
type _lig = Expect<Equal<L['$']['lights'], d.v4f[]>>;
type _keys = Expect<Equal<keyof L['entries'], 'camera' | 'particles' | 'lights' | 'albedo' | 'albedoSampler'>>;
export const _neg = () => {
  // @ts-expect-error -- lights must not be declared writable
  const _a: 'mutable' = sceneLayout.entries.lights.access;
};

assert.equal(sceneLayout.index, 2, 'layout index');
assert.equal(sceneLayout.resourceType, 'bind-group-layout');
assert.deepEqual(Object.keys(sceneLayout.entries), ['camera', 'particles', 'lights', 'albedo', 'albedoSampler'], 'entry order');

const wgsl = tgpu.resolve({
  template: `fn main() {
    let a = layout.$.camera;
    let b = layout.$.particles[0];
    let c = layout.$.lights[0];
    let t = textureSample(layout.$.albedo, layout.$.albedoSampler, vec2f());
  }`,
  externals: { layout: sceneLayout },
});
const decl = (binding: number, rest: string) =>
  new RegExp(`@group\\(\\s*2\\s*\\)\\s*@binding\\(\\s*${binding}\\s*\\)\\s*${rest}`);
assert.match(wgsl, decl(0, 'var<\\s*uniform\\s*>\\s*camera\\s*:\\s*Camera\\s*;'), 'binding 0 (uniform camera)\n' + wgsl);
assert.match(wgsl, decl(1, 'var<\\s*storage\\s*,\\s*read_write\\s*>\\s*particles\\s*:\\s*array<\\s*Particle\\s*>\\s*;'), 'binding 1 (read_write particles)\n' + wgsl);
assert.match(wgsl, decl(2, 'var<\\s*storage\\s*,\\s*read\\s*>\\s*lights\\s*:\\s*array<\\s*vec4f\\s*>\\s*;'), 'binding 2 (read lights)\n' + wgsl);
assert.match(wgsl, decl(3, 'var\\s+albedo\\s*:\\s*texture_2d<\\s*f32\\s*>\\s*;'), 'binding 3 (albedo texture)\n' + wgsl);
assert.match(wgsl, decl(4, 'var\\s+albedoSampler\\s*:\\s*sampler\\s*;'), 'binding 4 (sampler)\n' + wgsl);
assert.match(wgsl, /struct\s+Camera\s*\{\s*viewProj\s*:\s*mat4x4f\s*,\s*position\s*:\s*vec3f\s*,?\s*\}/, wgsl);
assert.match(wgsl, /struct\s+Particle\s*\{\s*pos\s*:\s*vec3f\s*,\s*vel\s*:\s*vec3f\s*,?\s*\}/, wgsl);
