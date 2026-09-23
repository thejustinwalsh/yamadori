// Leaves the u32 varying with default interpolation; WGSL requires @interpolate(flat) on integer vertex outputs and fragment inputs.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const mainVertex = tgpu.vertexFn({
  in: { position: d.location(0, d.vec2f), cellId: d.location(1, d.u32) },
  out: { pos: d.builtin.position, uv: d.vec2f, cell: d.u32 },
})`{
  return Out(vec4f(in.position, 0.0, 1.0), in.position * 0.5 + 0.5, in.cellId);
}`.$name('mainVertex');

export const mainFragment = tgpu.fragmentFn({
  in: { uv: d.vec2f, cell: d.u32 },
  out: d.vec4f,
})`{
  return vec4f(in.uv, f32(in.cell % 2u), 1.0);
}`.$name('mainFragment');
