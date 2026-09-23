// Uses an untyped slot instead of an accessor: a function bound to it is emitted as a bare identifier, not called.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const tintAccess = tgpu.slot(d.vec3f(1, 1, 1));

export const tinted = tgpu.fn([d.vec3f], d.vec3f)`(c: vec3f) -> vec3f { return c * tint; }`
  .$uses({ tint: tintAccess })
  .$name('tinted');

export function resolveTinted(color: d.v3f): string {
  return tgpu.resolve([tinted.with(tintAccess, color)]);
}
