// Uses raw WebGPU format strings as schema members; typegpu needs its vertex-format schemas (d.snorm8x4 etc.).
import * as d from 'typegpu/data';

export const PackedVertex = d.unstruct({
  position: 'float32x3',
  normal: 'snorm8x4',
  uv: 'unorm16x2',
  color: 'unorm8x4',
});

export const packedVertexStride: number = d.sizeOf(PackedVertex);
