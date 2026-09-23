// Passes the slot bindings as a map object to .with(); in 0.12.5 .with takes (slot, value).
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const reinhard = tgpu.fn([d.vec3f], d.vec3f)`(c: vec3f) -> vec3f { return c / (c + vec3f(1.0)); }`.$name('reinhard');
export const clampTonemap = tgpu.fn([d.vec3f], d.vec3f)`(c: vec3f) -> vec3f { return clamp(c, vec3f(0.0), vec3f(1.0)); }`.$name('clampTonemap');

export const tonemapSlot = tgpu.slot<typeof reinhard>();

export const shade = tgpu.fn([d.vec3f], d.vec4f)`(color: vec3f) -> vec4f { return vec4f(tonemap(color), 1.0); }`
  .$uses({ tonemap: tonemapSlot })
  .$name('shade');

export const shadeReinhard = shade.with({ slot: tonemapSlot, value: reinhard });
export const shadeClamp = shade.with({ slot: tonemapSlot, value: clampTonemap });
