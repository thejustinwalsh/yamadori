// Leaves out the initial values: seed and jitter are zero-initialised instead of 1 and (0.5, 1).
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const tile = tgpu.workgroupVar(d.arrayOf(d.f32, 256)).$name('tile');
export const seed = tgpu.privateVar(d.u32).$name('seed');
export const jitter = tgpu.privateVar(d.vec2f).$name('jitter');
