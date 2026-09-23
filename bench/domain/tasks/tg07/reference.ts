import * as d from 'typegpu/data';

export const PackedVertex = d.unstruct({
  position: d.float32x3,
  normal: d.snorm8x4,
  uv: d.unorm16x2,
  color: d.unorm8x4,
});

export const packedVertexStride: number = d.sizeOf(PackedVertex);
