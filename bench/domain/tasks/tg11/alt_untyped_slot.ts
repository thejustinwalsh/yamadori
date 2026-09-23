import { tgpu, d } from 'typegpu';

export const reinhard = tgpu.fn([d.vec3f], d.vec3f)`(c: vec3f) -> vec3f { return c / (c + vec3f(1.0)); }`.$name('reinhard');
export const clampTonemap = tgpu.fn([d.vec3f], d.vec3f)`(c: vec3f) -> vec3f { return clamp(c, vec3f(0.0), vec3f(1.0)); }`.$name('clampTonemap');

export const tonemapSlot = tgpu.slot<typeof reinhard>();

const body = '(color: vec3f) -> vec4f { let mapped = op(color); return vec4f(mapped, 1.0); }';
export const shade = tgpu.fn([d.vec3f], d.vec4f)(body).$uses({ op: tonemapSlot });
shade.$name('shade');

export const shadeReinhard = shade.with(tonemapSlot, reinhard);
export const shadeClamp = shade.with(tonemapSlot, clampTonemap);
