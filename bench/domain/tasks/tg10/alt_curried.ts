import { tgpu, d } from 'typegpu';

const { location, vec2f, vec4f, f32 } = d;

export const InstanceData = d.struct({
  offset: location(3, vec2f),
  scale: location(4, f32),
  color: location(5, vec4f),
});

export const instanceLayout = tgpu.vertexLayout(d.arrayOf(InstanceData), 'instance').$name('instances');
