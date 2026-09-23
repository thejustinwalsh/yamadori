// `tgpu.constant` is not an API in typegpu 0.12.5 (the function is `tgpu.const`).
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const GAUSS_WEIGHTS = tgpu
  .constant(d.arrayOf(d.f32, 5), [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216])
  .$name('GAUSS_WEIGHTS');
