// Alternative correct answer: the deprecated tgpu['~unstable'].derived alias still exists in 0.12.5.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const gridSize = tgpu.slot(16);

// The deprecated-but-present alias of tgpu.lazy.
export const cellSize = tgpu['~unstable'].derived(() => {
  const size = gridSize.$;
  return tgpu.const(d.f32, 1.0 / size).$name('CELL_SIZE');
});

export const cellOf = tgpu.fn([d.vec2f], d.vec2u)`(p: vec2f) -> vec2u { let c = CS; return vec2u(p / c); }`
  .$uses({ CS: cellSize })
  .$name('cellOf');
