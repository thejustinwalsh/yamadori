// Makes the shared tile a private (per-invocation) variable, so it is not shared across the workgroup.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const tile = tgpu.privateVar(d.arrayOf(d.f32, 256)).$name('tile');
export const seed = tgpu.privateVar(d.u32, 1).$name('seed');
export const jitter = tgpu.privateVar(d.vec2f, d.vec2f(0.5, 1)).$name('jitter');
