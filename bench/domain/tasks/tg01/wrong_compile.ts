// `d.sizeof` does not exist (it is `d.sizeOf`), and Infer is not on the default export.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, mass: d.f32 });
export type ParticleValue = tgpu.Infer<typeof Particle>;
export const particleByteSize: number = d.sizeof(Particle);
