import { tgpu, d, type TgpuRoot } from 'typegpu';

export const histLayout = tgpu.bindGroupLayout({
  values: { storage: d.arrayOf(d.f32) },
  bins: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
});

const WORKGROUP = 128;

export const histogram = tgpu
  .computeFn({ in: { id: d.builtin.globalInvocationId }, workgroupSize: [WORKGROUP] })(`{
    if (id.x >= arrayLength(&L.$.values)) { return; }
    let v = L.$.values[id.x];
    atomicAdd(&L.$.bins[min(u32(v * 16.0), 15u)], 1u);
  }`)
  .$uses({ L: histLayout })
  .$name('histogram');

// Fixed bindings instead of raw buffers, data uploaded after creation, and the
// (layout, group) form of .with().
export function runHistogram(root: TgpuRoot, values: number[]): void {
  const input = root.createReadonly(d.arrayOf(d.f32, values.length));
  input.write(values);
  const bins = root.createMutable(d.arrayOf(d.atomic(d.u32), 16));
  const group = root.createBindGroup(histLayout, { values: input, bins });
  root
    .createComputePipeline({ compute: histogram })
    .with(histLayout, group)
    .dispatchWorkgroups(Math.ceil(values.length / WORKGROUP));
}
