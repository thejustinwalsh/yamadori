import assert from 'node:assert/strict';
import tgpu, { writeToArrayBuffer } from 'typegpu';
import * as d from 'typegpu/data';
import type { Equal, Expect } from './assert_types.ts';
import { PackedVertex, packedVertexStride } from './solution.ts';

type _v = Expect<Equal<d.Infer<typeof PackedVertex>, { position: d.v3f; normal: d.v4f; uv: d.v2f; color: d.v4f }>>;
export const _neg = () => {
  // @ts-expect-error -- uv reads back as a vec2f, not a vec4f
  const _x: d.v4f = (null! as d.Infer<typeof PackedVertex>).uv;
};

assert.equal(d.sizeOf(PackedVertex), 24, 'record size');
assert.equal(packedVertexStride, 24, 'packedVertexStride');

// The formats and offsets WebGPU will see, read through a vertex layout.
const layout = tgpu.vertexLayout((n: number) => d.disarrayOf(PackedVertex, n));
assert.equal(layout.stride, 24, 'vertex layout stride');
const attrib = layout.attrib as Record<string, { format: string; offset: number }>;
const seen = Object.fromEntries(Object.entries(attrib).map(([k, v]) => [k, [v.format, v.offset]]));
assert.deepEqual(seen, {
  position: ['float32x3', 0],
  normal: ['snorm8x4', 12],
  uv: ['unorm16x2', 16],
  color: ['unorm8x4', 20],
});

// The bytes typegpu writes for one record.
const buf = new ArrayBuffer(24);
writeToArrayBuffer(buf, PackedVertex, {
  position: d.vec3f(1, 2, 3),
  normal: d.vec4f(1, -1, 0, 1),
  uv: d.vec2f(0, 1),
  color: d.vec4f(1, 0, 1, 0),
});
const dv = new DataView(buf);
assert.deepEqual([dv.getFloat32(0, true), dv.getFloat32(4, true), dv.getFloat32(8, true)], [1, 2, 3]);
assert.deepEqual([dv.getInt8(12), dv.getInt8(13), dv.getInt8(14), dv.getInt8(15)], [127, -127, 0, 127], 'snorm8x4');
assert.deepEqual([dv.getUint16(16, true), dv.getUint16(18, true)], [0, 65535], 'unorm16x2');
assert.deepEqual([...new Uint8Array(buf, 20, 4)], [255, 0, 255, 0], 'unorm8x4');
