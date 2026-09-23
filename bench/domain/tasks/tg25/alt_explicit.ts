import { tgpu, common, d, type TgpuRoot, type TgpuRenderPipeline } from 'typegpu';

export const blitLayout = tgpu.bindGroupLayout({
  src: { texture: d.texture2d(), visibility: ['fragment'] },
  srcSampler: { sampler: 'filtering', visibility: ['fragment'] },
});

export const blitFragment = tgpu
  .fragmentFn({ in: { uv: d.location(0, d.vec2f) }, out: d.location(0, d.vec4f) })(
    '{ let c = textureSample(L.$.src, L.$.srcSampler, in.uv); return c; }',
  )
  .$uses({ L: blitLayout });
blitFragment.$name('blit');

export const createBlitPipeline = (root: TgpuRoot, format: GPUTextureFormat): TgpuRenderPipeline<d.Vec4f> =>
  root.createRenderPipeline({ vertex: common.fullScreenTriangle, fragment: blitFragment, targets: { format } });

export function blit(root: TgpuRoot, pipeline: TgpuRenderPipeline<d.Vec4f>, source: GPUTextureView,
  sampler: GPUSampler, target: GPUTextureView): void {
  pipeline
    .with(blitLayout, root.createBindGroup(blitLayout, { src: source, srcSampler: sampler }))
    .withColorAttachment({ view: target, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' })
    .draw(3, 1);
}
