// Packs the floats tightly (13 per particle) into a Float32Array, ignoring WGSL vec3f alignment and struct padding.
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function packParticles(particles: d.Infer<typeof Particle>[]): ArrayBuffer {
  const out = new Float32Array(particles.length * 13);
  particles.forEach((p, i) => {
    out.set([...p.position, ...p.velocity, ...p.color, p.age], i * 13);
  });
  return out.buffer;
}
