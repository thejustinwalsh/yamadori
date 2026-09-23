// Never uploads the input: the values buffer is created without initial data, so the GPU sees zeros.
import tgpu, { type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';

export const histLayout = tgpu.bindGroupLayout({
  values: { storage: (n: number) => d.arrayOf(d.f32, n), access: 'readonly' },
  bins: { storage: (n: number) => d.arrayOf(d.atomic(d.u32), n), access: 'mutable' },
});

export const histogram = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [128],
})`{
  let i = in.gid.x;
  if (i >= arrayLength(&layout.$.values)) {
    return;
  }
  let bin = min(u32(layout.$.values[i] * 16.0), 15u);
  atomicAdd(&layout.$.bins[bin], 1u);
}`
  .$uses({ layout: histLayout })
  .$name('histogram');

export function runHistogram(root: TgpuRoot, values: number[]): void {
  const valuesBuffer = root.createBuffer(d.arrayOf(d.f32, values.length)).$usage('storage');
  const binsBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), 16)).$usage('storage');
  const bindGroup = root.createBindGroup(histLayout, { values: valuesBuffer, bins: binsBuffer });
  const pipeline = root.createComputePipeline({ compute: histogram });
  pipeline.with(bindGroup).dispatchWorkgroups(Math.ceil(values.length / 128));
}
