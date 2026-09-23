import * as d from 'typegpu/data';

export const Particle = d.struct({ position: d.vec3f, velocity: d.vec3f, color: d.vec4f, age: d.f32 });

// Hand-rolled, but with offsets taken from typegpu's layout helper.
const at = {
  position: d.memoryLayoutOf(Particle, (p) => p.position).offset,
  velocity: d.memoryLayoutOf(Particle, (p) => p.velocity).offset,
  color: d.memoryLayoutOf(Particle, (p) => p.color).offset,
  age: d.memoryLayoutOf(Particle, (p) => p.age).offset,
};
const stride = d.sizeOf(Particle);

export function packParticles(particles: d.Infer<typeof Particle>[]): ArrayBuffer {
  const buf = new ArrayBuffer(stride * particles.length);
  const view = new DataView(buf);
  particles.forEach((p, i) => {
    const base = i * stride;
    const put = (off: number, xs: ArrayLike<number>) => {
      for (let k = 0; k < xs.length; k++) view.setFloat32(base + off + 4 * k, xs[k]!, true);
    };
    put(at.position, [p.position.x, p.position.y, p.position.z]);
    put(at.velocity, [p.velocity.x, p.velocity.y, p.velocity.z]);
    put(at.color, [p.color.x, p.color.y, p.color.z, p.color.w]);
    view.setFloat32(base + at.age, p.age, true);
  });
  return buf;
}
