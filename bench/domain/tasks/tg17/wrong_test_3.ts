// Rewrites the whole particle with writeToArrayBuffer, clobbering its other fields (and needs values it does not have).
import { readFromArrayBuffer, writeToArrayBuffer } from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function setVelocity(buffer: ArrayBuffer, count: number, index: number, velocity: d.v3f): void {
  const schema = d.arrayOf(Particle, count);
  const all = readFromArrayBuffer(buffer, schema);
  all[index] = { position: d.vec3f(), velocity, color: d.vec4f(), age: 0 };
  writeToArrayBuffer(buffer, schema, all);
}
