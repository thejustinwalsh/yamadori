// Allocates only one element's worth of bytes (d.sizeOf(Particle)) instead of the whole array.
import { writeToArrayBuffer } from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function packParticles(particles: d.Infer<typeof Particle>[]): ArrayBuffer {
  const buffer = new ArrayBuffer(d.sizeOf(Particle));
  writeToArrayBuffer(buffer, d.arrayOf(Particle, particles.length), particles);
  return buffer;
}
