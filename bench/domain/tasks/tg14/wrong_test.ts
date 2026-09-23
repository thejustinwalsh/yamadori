// A GPU function (tgpu.fn with a JS body) is translated to WGSL, not evaluated on the CPU at shader-generation time.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const hexToRgb = tgpu.fn([d.u32], d.vec3f)((hex) => {
  'use gpu';
  const r = (hex >> 16) & 0xff;
  const g = (hex >> 8) & 0xff;
  const b = hex & 0xff;
  return d.vec3f(r / 255, g / 255, b / 255);
});
