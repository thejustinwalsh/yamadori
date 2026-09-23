// Reads gridSize.$ at module load; a slot's value is only available during shader generation, so importing the module throws.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const gridSize = tgpu.slot<number>(16);

export const cellSize = tgpu.const(d.f32, 1 / gridSize.$).$name('CELL_SIZE');

export const cellOf = tgpu.fn([d.vec2f], d.vec2u)`(p: vec2f) -> vec2u { return vec2u(p / cell); }`
  .$uses({ cell: cellSize })
  .$name('cellOf');
