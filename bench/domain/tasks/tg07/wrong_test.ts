// A WGSL struct of float vectors: aligned to 16 and 32-bit components, so the record is far larger than 24 bytes.
import * as d from 'typegpu/data';

export const PackedVertex = d.struct({
  position: d.vec3f,
  normal: d.vec4f,
  uv: d.vec2f,
  color: d.vec4f,
});

export const packedVertexStride: number = d.sizeOf(PackedVertex);
