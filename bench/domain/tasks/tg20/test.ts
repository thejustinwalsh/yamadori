import assert from 'node:assert/strict';
import tgpu, {
  writeToArrayBuffer,
  type StorageFlag, type TgpuBuffer, type TgpuMutable, type TgpuReadonly, type TgpuUniform, type UniformFlag, type VertexFlag,
} from 'typegpu';
import * as d from 'typegpu/data';
import type { IsAny, Expect, Equal } from './assert_types.ts';
import { Counters, createSimBuffers, Particle, SimParams } from './solution.ts';

type R = ReturnType<typeof createSimBuffers>;
type _notAny = Expect<Equal<IsAny<R>, false>>;
const _types = (r: R) => {
  const _p: TgpuUniform<typeof SimParams> = r.params;
  const _m: TgpuMutable<d.WgslArray<typeof Particle>> = r.particles;
  const _c: TgpuBuffer<typeof Counters> & StorageFlag = r.counters;
  const _q: TgpuBuffer<d.WgslArray<d.Vec2f>> & VertexFlag = r.quad;
  // @ts-expect-error -- params is a uniform binding, not a writable storage binding
  const _pm: TgpuMutable<typeof SimParams> = r.params;
  // @ts-expect-error -- particles must not be read-only
  const _mr: TgpuReadonly<d.WgslArray<typeof Particle>> = r.particles;
  // @ts-expect-error -- counters is not flagged for uniform use
  const _cu: UniformFlag = r.counters;
};
void _types;

// A root over a stub device. typegpu creates GPU buffers lazily, so none of
// this touches the device; any call into it throws and fails the test.
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
  UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
const device = new Proxy({ features: new Set(), limits: {}, lost: new Promise(() => {}), queue: {} } as any, {
  get(t, k) { return k in t ? t[k] : () => { throw new Error(`stub device.${String(k)} called`); }; },
}) as GPUDevice;
const root = tgpu.initFromDevice({ device });
const r = createSimBuffers(root);

assert.equal(r.params.resourceType, 'uniform', 'params is a fixed uniform binding');
assert.equal(r.particles.resourceType, 'mutable', 'particles is a fixed mutable binding');
assert.equal(r.counters.resourceType, 'buffer', 'counters is a plain buffer');
assert.equal(r.quad.resourceType, 'buffer', 'quad is a plain buffer');
assert.equal(r.counters.usableAsStorage, true, 'counters flagged for storage');
assert.equal(r.quad.usableAsVertex, true, 'quad flagged as vertex');

// Initial contents, normalised through typegpu's serializer so any accepted
// input form (vectors, tuples, typed arrays, an init callback) is handled.
function initialBytes(buffer: { initial?: unknown; dataType: d.AnyWgslData }): Float32Array {
  let data: unknown = buffer.initial;
  if (typeof data === 'function') {
    let captured: unknown;
    (data as (b: unknown) => void)({ write: (x: unknown) => { captured = x; } });
    data = captured;
  }
  assert.ok(data !== undefined, 'buffer has initial data');
  const out = new ArrayBuffer(d.sizeOf(buffer.dataType));
  writeToArrayBuffer(out, buffer.dataType, data as never);
  return new Float32Array(out);
}
const pf = initialBytes(r.params.buffer as never);
assert.ok(Math.abs(pf[0]! - 1 / 60) < 1e-7, `dt = ${pf[0]}`);
assert.deepEqual([pf[4], pf[6]], [0, 0], 'gravity.x / gravity.z');
assert.ok(Math.abs(pf[5]! + 9.81) < 1e-5, `gravity.y = ${pf[5]}`);

const qf = initialBytes(r.quad as never);
const vert = (i: number) => `${qf[2 * i]},${qf[2 * i + 1]}`;
const tri = (i: number) => [vert(i), vert(i + 1), vert(i + 2)].sort().join(' ');
assert.deepEqual([tri(0), tri(3)].sort(), ['-1,-1 -1,1 1,-1', '-1,1 1,-1 1,1'].sort(), 'two triangles covering clip space');

const wgsl = tgpu.resolve({
  template: 'fn f() { let a = P.dt; let b = PS[0]; let c = C.alive; }',
  externals: { P: r.params, PS: r.particles, C: r.counters.as('mutable') },
});
assert.match(wgsl, /var<\s*uniform\s*>\s*P\s*:\s*SimParams\s*;/, wgsl);
assert.match(wgsl, /var<\s*storage\s*,\s*read_write\s*>\s*PS\s*:\s*array<\s*Particle\s*,\s*1024\s*>\s*;/, wgsl);
assert.match(wgsl, /struct\s+Counters\s*\{\s*alive\s*:\s*atomic<u32>/, wgsl);
