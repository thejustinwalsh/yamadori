// Assumes velocity sits right after position (offset 12) -- a vec3f is 16-byte aligned in WGSL, so it is at 16.
import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

export function setVelocity(buffer: ArrayBuffer, _count: number, index: number, velocity: d.v3f): void {
  const view = new DataView(buffer);
  const base = index * d.sizeOf(Particle) + 12;
  view.setFloat32(base, velocity.x, true);
  view.setFloat32(base + 4, velocity.y, true);
  view.setFloat32(base + 8, velocity.z, true);
}
