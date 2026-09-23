import assert from 'node:assert/strict';
import type { TgpuVertexLayout } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { InstanceData, instanceLayout } from './solution.ts';

type _v = Expect<Equal<d.Infer<typeof InstanceData>, { offset: d.v2f; scale: number; color: d.v4f }>>;
const _isLayout: TgpuVertexLayout = instanceLayout;
export const _neg = () => {
  // @ts-expect-error -- a vertex layout is not a raw GPUVertexBufferLayout
  const _raw: GPUVertexBufferLayout = instanceLayout;
};

assert.equal(d.sizeOf(InstanceData), 32, 'WGSL-aligned struct: vec2f@0, f32@8, vec4f@16 -> 32');
assert.equal(instanceLayout.stride, 32, 'stride');
assert.equal(instanceLayout.stepMode, 'instance', 'step mode');

let raw: GPUVertexBufferLayout;
try {
  raw = instanceLayout.vertexLayout;
} catch (e) {
  throw new Error('instanceLayout.vertexLayout threw: ' + (e as Error).message);
}
assert.deepEqual(JSON.parse(JSON.stringify(raw)), {
  arrayStride: 32,
  stepMode: 'instance',
  attributes: [
    { format: 'float32x2', offset: 0, shaderLocation: 3 },
    { format: 'float32', offset: 8, shaderLocation: 4 },
    { format: 'float32x4', offset: 16, shaderLocation: 5 },
  ],
});
