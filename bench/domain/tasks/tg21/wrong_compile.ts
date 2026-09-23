// Uses the positional (inputs, outputs) vertexFn/fragmentFn signature of older typegpu; 0.12.5 takes one { in, out } options object.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const mainVertex = tgpu.vertexFn(
  { position: d.location(0, d.vec2f), cellId: d.location(1, d.u32) },
  { pos: d.builtin.position, uv: d.vec2f, cell: d.interpolate('flat', d.u32) },
)`{
  return Out(vec4f(in.position, 0.0, 1.0), in.position * 0.5 + 0.5, in.cellId);
}`.$name('mainVertex');

export const mainFragment = tgpu.fragmentFn(
  { uv: d.vec2f, cell: d.interpolate('flat', d.u32) },
  d.vec4f,
)`{
  return vec4f(in.uv, f32(in.cell % 2u), 1.0);
}`.$name('mainFragment');
