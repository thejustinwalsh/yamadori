// Forgets the step mode, so the buffer advances per vertex (the default) instead of per instance.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const InstanceData = d.struct({
  offset: d.location(3, d.vec2f),
  scale: d.location(4, d.f32),
  color: d.location(5, d.vec4f),
});

export const instanceLayout = tgpu.vertexLayout((n: number) => d.arrayOf(InstanceData, n));
