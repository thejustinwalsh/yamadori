// Draws six vertices as if the vertex stage were a two-triangle quad; the built-in full-screen triangle uses three.
import tgpu, { common, type TgpuRoot } from 'typegpu';
import * as d from 'typegpu/data';

export const blitLayout = tgpu.bindGroupLayout({
  src: { texture: d.texture2d(d.f32) },
  srcSampler: { sampler: 'filtering' },
});

export const blitFragment = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })`{
  return textureSample(layout.$.src, layout.$.srcSampler, in.uv);
}`
  .$uses({ layout: blitLayout })
  .$name('blit');

export function createBlitPipeline(root: TgpuRoot, format: GPUTextureFormat) {
  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: blitFragment,
    targets: { format },
  });
}

export function blit(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createBlitPipeline>,
  source: GPUTextureView,
  sampler: GPUSampler,
  target: GPUTextureView,
): void {
  const bindGroup = root.createBindGroup(blitLayout, { src: source, srcSampler: sampler });
  pipeline
    .withColorAttachment({ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] })
    .with(bindGroup)
    .draw(6);
}
