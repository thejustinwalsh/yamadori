// Uses tgpu.derived (the pre-stable name); in 0.12.5 it survives only as tgpu['~unstable'].derived, the stable API is tgpu.lazy.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const gridSize = tgpu.slot<number>(16);

export const cellSize = tgpu.derived(() => tgpu.const(d.f32, 1 / gridSize.$).$name('CELL_SIZE'));

export const cellOf = tgpu.fn([d.vec2f], d.vec2u)`(p: vec2f) -> vec2u { return vec2u(p / cell); }`
  .$uses({ cell: cellSize })
  .$name('cellOf');
