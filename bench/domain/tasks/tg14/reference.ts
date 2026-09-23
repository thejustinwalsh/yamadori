import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const hexToRgb = tgpu.comptime((hex: number): d.v3f => {
  const r = (hex >> 16) & 0xff;
  const g = (hex >> 8) & 0xff;
  const b = hex & 0xff;
  return d.vec3f(r / 255, g / 255, b / 255);
});
