import { tgpu } from 'typegpu';
import { arrayOf, f32 } from 'typegpu/data';

const weights = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216];

// The runtime-sized array constructor form: tgpu.const also accepts `(n) => WgslArray`.
export const GAUSS_WEIGHTS = tgpu.const(arrayOf(f32), weights);
GAUSS_WEIGHTS.$name('GAUSS_WEIGHTS');
