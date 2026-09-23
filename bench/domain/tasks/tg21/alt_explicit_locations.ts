import { tgpu, d } from 'typegpu';

// Shared varyings with explicit, non-default locations on both sides.
const varyings = {
  uv: d.location(2, d.interpolate('perspective', d.vec2f)),
  cell: d.location(5, d.interpolate('flat', d.u32)),
};

export const mainVertex = tgpu
  .vertexFn({
    in: { position: d.location(0, d.vec2f), cellId: d.location(1, d.u32) },
    out: { pos: d.builtin.position, ...varyings },
  })(`{
    let p = in.position;
    return Out(vec4f(p, 0.0, 1.0), p * 0.5 + 0.5, in.cellId);
  }`)
  .$name('mainVertex');

export const mainFragment = tgpu
  .fragmentFn({ in: varyings, out: d.location(0, d.vec4f) })(`{
    let parity = f32(in.cell % 2u);
    return vec4f(in.uv, parity, 1.0);
  }`)
  .$name('mainFragment');
