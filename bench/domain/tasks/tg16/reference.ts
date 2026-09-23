import { writeToArrayBuffer } from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function packParticles(particles: d.Infer<typeof Particle>[]): ArrayBuffer {
  const schema = d.arrayOf(Particle, particles.length);
  const buffer = new ArrayBuffer(d.sizeOf(schema));
  writeToArrayBuffer(buffer, schema, particles);
  return buffer;
}
