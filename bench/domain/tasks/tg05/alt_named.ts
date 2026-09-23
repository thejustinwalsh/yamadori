import { tgpu, d } from 'typegpu';

const tile = tgpu.workgroupVar(d.arrayOf(d.f32, 256));
tile.$name('tile');
const seed = tgpu.privateVar(d.u32, 1);
seed.$name('seed');
const jitter = tgpu.privateVar(d.vec2f, d.vec2f(0.5, 1.0));
jitter.$name('jitter');

export { tile, seed, jitter };
