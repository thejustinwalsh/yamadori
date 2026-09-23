import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const gridSize = tgpu.slot<number>(16);

export const cellSize = tgpu.lazy(() => tgpu.const(d.f32, 1 / gridSize.$).$name('CELL_SIZE'));

export const cellOf = tgpu.fn([d.vec2f], d.vec2u)`(p: vec2f) -> vec2u { return vec2u(p / cell); }`
  .$uses({ cell: cellSize })
  .$name('cellOf');
