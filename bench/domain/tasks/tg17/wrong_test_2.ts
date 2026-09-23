// Patches against the element schema instead of the array schema, so it always writes particle 0.
import { patchArrayBuffer } from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function setVelocity(buffer: ArrayBuffer, _count: number, _index: number, velocity: d.v3f): void {
  patchArrayBuffer(buffer, Particle, { velocity });
}
