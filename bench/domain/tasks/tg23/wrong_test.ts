// Computes the constant once, eagerly, from the slot default -- overriding gridSize with .with() no longer changes it.
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

export const gridSize = tgpu.slot<number>(16);

export const cellSize = tgpu.const(d.f32, 1 / (gridSize.defaultValue ?? 16)).$name('CELL_SIZE');

export const cellOf = tgpu.fn([d.vec2f], d.vec2u)`(p: vec2f) -> vec2u { return vec2u(p / cell); }`
  .$uses({ cell: cellSize })
  .$name('cellOf');
