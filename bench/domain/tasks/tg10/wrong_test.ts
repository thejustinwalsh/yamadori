// Plain struct without @location attributes: typegpu cannot produce a raw GPUVertexBufferLayout (no shader locations).
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const InstanceData = d.struct({
  offset: d.vec2f,
  scale: d.f32,
  color: d.vec4f,
});

export const instanceLayout = tgpu.vertexLayout((n: number) => d.arrayOf(InstanceData, n), 'instance');
