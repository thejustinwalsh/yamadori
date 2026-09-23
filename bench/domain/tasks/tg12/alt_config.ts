import { tgpu, d } from 'typegpu';

export const tintAccess = tgpu.accessor(d.vec3f, d.vec3f(1));

export const tinted = tgpu.fn([d.vec3f], d.vec3f)(
  '(c: vec3f) -> vec3f { let t = TINT; return c * t; }',
).$uses({ TINT: tintAccess });
tinted.$name('tinted');

// Bind through the resolve options instead of .with() on the function.
export const resolveTinted = (color: d.v3f): string =>
  tgpu.resolve([tinted], { config: (cfg) => cfg.with(tintAccess, color) });
