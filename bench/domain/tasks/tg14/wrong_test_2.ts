// Scales by 256 instead of 255, so full-intensity channels come out below 1.0.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const hexToRgb = tgpu.comptime((hex: number) =>
  d.vec3f(((hex >> 16) & 0xff) / 256, ((hex >> 8) & 0xff) / 256, (hex & 0xff) / 256));
