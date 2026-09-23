// There is no generic d.ptr(addressSpace, schema) constructor in 0.12.5 (it is d.ptrFn / d.ptrPrivate / ...).
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const accumulate = tgpu.fn([d.ptr('function', d.vec3f), d.vec3f, d.f32])`(acc: ptr<function, vec3f>, sample: vec3f, weight: f32) {
  *acc += sample * weight;
}`.$name('accumulate');

export const bumpCounter = tgpu.fn([d.ptr('private', d.u32)])`(counter: ptr<private, u32>) {
  *counter += 1u;
}`.$name('bumpCounter');
