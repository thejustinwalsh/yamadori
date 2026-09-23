// Passes a raw WebGPU GPUVertexBufferLayout descriptor; tgpu.vertexLayout takes a (count) => array schema constructor.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const InstanceData = d.struct({ offset: d.vec2f, scale: d.f32, color: d.vec4f });

export const instanceLayout = tgpu.vertexLayout({
  arrayStride: 32,
  stepMode: 'instance',
  attributes: [
    { format: 'float32x2', offset: 0, shaderLocation: 3 },
    { format: 'float32', offset: 8, shaderLocation: 4 },
    { format: 'float32x4', offset: 16, shaderLocation: 5 },
  ],
});
