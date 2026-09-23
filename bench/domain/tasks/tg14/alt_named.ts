import { tgpu, d } from 'typegpu';

function unpack(hex: number) {
  return d.vec3f(((hex >>> 16) % 256) / 255, (Math.floor(hex / 256) % 256) / 255, (hex % 256) / 255);
}

export const hexToRgb = tgpu.comptime(unpack).$name('hexToRgb');
