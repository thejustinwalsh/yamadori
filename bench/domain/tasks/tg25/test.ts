import assert from 'node:assert/strict';
import tgpu, { type TgpuBindGroupLayout, type TgpuFragmentFn, type TgpuRenderPipeline, type TgpuRoot } from 'typegpu';
import type { Equal, Expect, IsAny } from './assert_types.ts';
import { blit, blitFragment, blitLayout, createBlitPipeline } from './solution.ts';

const _l: TgpuBindGroupLayout = blitLayout;
const _f: TgpuFragmentFn = blitFragment as TgpuFragmentFn<any, any>;
type P = ReturnType<typeof createBlitPipeline>;
type _p = Expect<Equal<IsAny<P>, false>>;
const _pp = (p: P): TgpuRenderPipeline<any> => p;
type _keys = Expect<Equal<keyof (typeof blitLayout)['entries'], 'src' | 'srcSampler'>>;
export const _neg = () => {
  // @ts-expect-error -- the format is a GPUTextureFormat
  createBlitPipeline(null! as TgpuRoot, 'not-a-format');
};

// ---- a recording stub GPUDevice ------------------------------------------
// No GPU in node. typegpu only talks to the device through these calls, so
// recording them shows exactly what a real device would be asked to do.
const g = globalThis as any;
g.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
g.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
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
const device = new Proxy({ features: new Set(), limits: {}, lost: new Promise(() => {}), queue: recorder('queue') } as any, {
  get(t, k) {
    if (k in t) return t[k];
    if (typeof k !== 'string' || !k.startsWith('create')) return undefined;
    return (desc: any) => {
      const result = recorder(k.slice(6));
      calls.push({ what: `device.${k}`, args: [desc], result });
      return result;
    };
  },
}) as GPUDevice;
const root = tgpu.initFromDevice({ device });
const find = (what: string) => calls.filter((c) => c.what.endsWith(what));

// ---- run it ---------------------------------------------------------------
const srcView = { label: 'src-view' } as unknown as GPUTextureView;
const sampler = { label: 'linear' } as unknown as GPUSampler;
const dstView = { label: 'dst-view' } as unknown as GPUTextureView;

const pipeline = createBlitPipeline(root, 'rgba16float');
blit(root, pipeline, srcView, sampler, dstView);

const rp = find('device.createRenderPipeline');
assert.equal(rp.length, 1, 'one render pipeline');
const desc = rp[0]!.args[0];
assert.deepEqual(desc.fragment?.targets?.map((t: any) => t.format), ['rgba16float'], 'one colour target in the requested format');
assert.equal((desc.vertex?.buffers ?? []).length, 0, 'no vertex buffers: the full-screen triangle is generated from vertex_index');

const code: string = find('device.createShaderModule').map((c) => c.args[0].code).join('\n');
// typegpu's built-in full-screen triangle (common/fullScreenTriangle.js).
assert.match(code, /array<\s*vec2f\s*,\s*3\s*>\s*\(\s*vec2f\(\s*-1\s*,\s*-1\s*\)\s*,\s*vec2f\(\s*3\s*,\s*-1\s*\)\s*,\s*vec2f\(\s*-1\s*,\s*3\s*\)\s*\)/,
  'vertex stage is the built-in full-screen triangle\n' + code);
assert.match(code, /@fragment\s+fn\s+blit\s*\(/, code);
assert.match(code, /textureSample\s*\(\s*src\s*,\s*srcSampler\s*,/, code);
assert.match(code, /var\s+src\s*:\s*texture_2d<\s*f32\s*>/, code);
assert.match(code, /var\s+srcSampler\s*:\s*sampler\s*;/, code);

const bgl = find('device.createBindGroupLayout').map((c) => c.args[0].entries);
assert.ok(bgl.some((es: any[]) => es.length === 2 && es[0].binding === 0 && es[0].texture?.sampleType === 'float'
  && es[1].binding === 1 && es[1].sampler?.type === 'filtering'), 'bind group layout: float texture + filtering sampler');

const passes = find('.beginRenderPass');
assert.equal(passes.length, 1, 'one render pass');
const att = passes[0]!.args[0].colorAttachments;
assert.equal(att.length, 1);
assert.equal(att[0].view, dstView, 'renders into the target view');
assert.equal(att[0].loadOp, 'clear', 'clears the target');
assert.equal(att[0].storeOp, 'store');
const cv = att[0].clearValue;
assert.deepEqual(Array.isArray(cv) ? [...cv] : [cv?.r, cv?.g, cv?.b, cv?.a], [0, 0, 0, 0], 'transparent black');

const bgs = find('device.createBindGroup');
const setBG = find('.setBindGroup');
assert.equal(setBG.length, 1, 'one bind group set');
const bg = bgs.find((c) => c.result === setBG[0]!.args[1]);
assert.ok(bg, 'the bound group was created for the blit layout');
const res = new Map<number, unknown>(bg.args[0].entries.map((e: any) => [e.binding, e.resource]));
assert.equal(res.get(0), srcView, 'binding 0 is the source view');
assert.equal(res.get(1), sampler, 'binding 1 is the sampler');

const draws = find('.draw');
assert.equal(draws.length, 1, 'one draw');
assert.equal(draws[0]!.args[0], 3, 'three vertices');
assert.ok((draws[0]!.args[1] ?? 1) === 1, 'one instance');
assert.equal(find('queue.submit').length, 1, 'submitted');

// A second pipeline honours its own format.
root.unwrap(createBlitPipeline(root, 'r8unorm'));
assert.deepEqual(find('device.createRenderPipeline').at(-1)!.args[0].fragment.targets.map((t: any) => t.format), ['r8unorm'],
  'format is taken from the argument');
