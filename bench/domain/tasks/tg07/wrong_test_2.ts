// Packs the record (unstruct) but keeps 32-bit float vectors instead of the normalized 8/16-bit vertex formats.
import * as d from 'typegpu/data';

export const PackedVertex = d.unstruct({
  position: d.vec3f,
  normal: d.vec4f,
  uv: d.vec2f,
  color: d.vec4f,
});

export const packedVertexStride: number = d.sizeOf(PackedVertex);
