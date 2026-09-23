import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function setVelocity(buffer: ArrayBuffer, count: number, index: number, velocity: d.v3f): void {
  const { offset } = d.memoryLayoutOf(d.arrayOf(Particle, count), (ps) => ps[index]!.velocity);
  new Float32Array(buffer, offset, 3).set([velocity.x, velocity.y, velocity.z]);
}
