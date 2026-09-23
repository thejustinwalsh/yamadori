// Derives a plain number instead of a named constant, so the value is inlined as a literal and no CELL_SIZE constant is emitted.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const gridSize = tgpu.slot<number>(16);

export const cellSize = tgpu.lazy(() => 1 / gridSize.$);

export const cellOf = tgpu.fn([d.vec2f], d.vec2u)`(p: vec2f) -> vec2u { return vec2u(p / cell); }`
  .$uses({ cell: cellSize })
  .$name('cellOf');
