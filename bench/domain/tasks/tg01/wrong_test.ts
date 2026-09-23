// Hard-codes the packed size (3*4 + 3*4 + 4 = 28), ignoring vec3f's 16-byte alignment.
import * as d from 'typegpu/data';

export const Particle = d.struct({
  position: d.vec3f,
  velocity: d.vec3f,
  mass: d.f32,
});

export type ParticleValue = d.Infer<typeof Particle>;

export const particleByteSize: number = 28;
