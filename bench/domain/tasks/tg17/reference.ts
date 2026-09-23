import { patchArrayBuffer } from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function setVelocity(buffer: ArrayBuffer, count: number, index: number, velocity: d.v3f): void {
  patchArrayBuffer(buffer, d.arrayOf(Particle, count), { [index]: { velocity } });
}
