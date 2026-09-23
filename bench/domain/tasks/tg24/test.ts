import assert from 'node:assert/strict';
import tgpu, { type TgpuBindGroupLayout, type TgpuComputeFn, type TgpuRoot } from 'typegpu';
import type { Equal, Expect } from './assert_types.ts';
import { histLayout, histogram, runHistogram } from './solution.ts';

const _l: TgpuBindGroupLayout = histLayout;
const _c: TgpuComputeFn = histogram;
type _r = Expect<Equal<Parameters<typeof runHistogram>, [root: TgpuRoot, values: number[]]>>;
type _keys = Expect<Equal<keyof (typeof histLayout)['entries'], 'values' | 'bins'>>;
export const _neg = () => {
  // @ts-expect-error -- values are numbers
  runHistogram(null! as TgpuRoot, ['0.5']);
};

// ---- a recording stub GPUDevice ------------------------------------------
// No GPU in node. typegpu only talks to the device through these calls, so
// recording them shows exactly what a real device would be asked to do.
const g = globalThis as any;
g.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
g.GPUMapMode ??= { READ: 1, WRITE: 2 };
type Call = { what: string; args: any[]; result?: any };
const calls: Call[] = [];
let nextId = 0;
function recorder(kind: string): any {
  const target: any = { __id: `${kind}#${++nextId}` };
  return new Proxy(target, {
    get(t, k) {
      if (k in t) return t[k];
      if (k === 'then' || k === 'toJSON' || typeof k === 'symbol') return undefined;
      return (...args: any[]) => {
        const result = k.startsWith('begin') ? recorder('pass') : k === 'finish' ? recorder('commands')
          : k === 'getBindGroupLayout' ? recorder('bgl') : undefined;
        calls.push({ what: `${kind}.${k}`, args, result });
        return result;
      };
    },
  });
}
function gpuBuffer(desc: GPUBufferDescriptor): any {
  const b: any = { __id: `buffer#${++nextId}`, size: desc.size, usage: desc.usage, label: desc.label,
    mapState: desc.mappedAtCreation ? 'mapped' : 'unmapped', data: new ArrayBuffer(desc.size) };
  b.getMappedRange = () => b.data;
  b.unmap = () => { b.mapState = 'unmapped'; };
  b.destroy = () => {};
  return b;
}
// queue.writeBuffer really writes into the stub buffer, so uploads done after
// creation are visible too.
const queue = recorder('queue');
const queueWrites = (buffer: any, offset: number, data: BufferSource, dataOffset = 0, size?: number) => {
  calls.push({ what: 'queue.writeBuffer', args: [buffer, offset, data, dataOffset, size] });
  const bytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data as ArrayBuffer);
  const unit = ArrayBuffer.isView(data) && 'BYTES_PER_ELEMENT' in data ? (data as any).BYTES_PER_ELEMENT : 1;
  const src = bytes.subarray(dataOffset * unit, size === undefined ? undefined : (dataOffset + size) * unit);
  new Uint8Array(buffer.data).set(src, offset);
};
const device = new Proxy({
  features: new Set(), limits: {}, lost: new Promise(() => {}),
  queue: new Proxy({}, { get: (_t, k) => (k === 'writeBuffer' ? queueWrites : (queue as any)[k]) }),
} as any, {
  get(t, k) {
    if (k in t) return t[k];
    if (typeof k !== 'string' || !k.startsWith('create')) return undefined;
    return (desc: any) => {
      const result = k === 'createBuffer' ? gpuBuffer(desc) : recorder(k.slice(6));
      calls.push({ what: `device.${k}`, args: [desc], result });
      return result;
    };
  },
}) as GPUDevice;
const root = tgpu.initFromDevice({ device });
const find = (what: string) => calls.filter((c) => c.what.endsWith(what));

// ---- run it ---------------------------------------------------------------
const values = Array.from({ length: 300 }, (_, i) => (i % 16) / 16 + 0.01);
runHistogram(root, values);

const dispatches = find('.dispatchWorkgroups');
assert.equal(dispatches.length, 1, 'one dispatch');
const [x, y, z] = dispatches[0]!.args;
assert.equal(x, 3, 'ceil(300 / 128) = 3 workgroups, not one per element');
assert.ok((y ?? 1) === 1 && (z ?? 1) === 1, 'one-dimensional dispatch');
assert.equal(find('queue.submit').length, 1, 'the work is submitted');

const pipelines = find('device.createComputePipeline');
assert.equal(pipelines.length, 1, 'one compute pipeline');
const modules = find('device.createShaderModule');
assert.equal(modules.length, 1, 'one shader module');
const code: string = modules[0]!.args[0].code;
assert.match(code, /@compute\s+@workgroup_size\(\s*128\s*(?:,\s*1\s*){0,2}\)/, code);
assert.match(code, /@binding\(\s*0\s*\)\s*var<\s*storage\s*,\s*read\s*>\s*values\s*:\s*array<\s*f32\s*>/, code);
assert.match(code, /@binding\(\s*1\s*\)\s*var<\s*storage\s*,\s*read_write\s*>\s*bins\s*:\s*array<\s*atomic<\s*u32\s*>\s*>/, code);
assert.match(code, /atomicAdd\s*\(\s*&\s*bins\s*\[/, code);
assert.match(code, /arrayLength\s*\(\s*&\s*values\s*\)/, code);

const bglEntries = find('device.createBindGroupLayout')[0]?.args[0].entries;
assert.deepEqual(bglEntries?.map((e: any) => [e.binding, e.buffer?.type]), [[0, 'read-only-storage'], [1, 'storage']], 'bind group layout entries');

const setBG = find('.setBindGroup');
assert.equal(setBG.length, 1, 'one bind group set');
assert.equal(setBG[0]!.args[0], 0, 'at group index 0');
const bg = find('device.createBindGroup').find((c) => c.result === setBG[0]!.args[1]);
assert.ok(bg, 'the bound group is the one created for histLayout');
const byBinding = new Map<number, any>(bg.args[0].entries.map((e: any) => [e.binding, e.resource.buffer]));
const valuesGpu = byBinding.get(0), binsGpu = byBinding.get(1);
assert.equal(valuesGpu?.size, 1200, 'values buffer holds 300 f32');
assert.equal(binsGpu?.size, 64, 'bins buffer holds 16 atomic u32');
assert.ok(valuesGpu.usage & g.GPUBufferUsage.STORAGE, 'values buffer has STORAGE usage');
assert.ok(binsGpu.usage & g.GPUBufferUsage.STORAGE, 'bins buffer has STORAGE usage');
const uploaded = [...new Float32Array(valuesGpu.data)];
assert.equal(uploaded.length, 300);
uploaded.forEach((v, i) => assert.ok(Math.abs(v - values[i]!) < 1e-6, `values[${i}] uploaded`));
