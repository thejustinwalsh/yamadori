import { float32x3, sizeOf, snorm8x4, unorm16x2, unorm8x4, unstruct } from 'typegpu/data';

const attributes = { position: float32x3, normal: snorm8x4, uv: unorm16x2, color: unorm8x4 };
export const PackedVertex = unstruct(attributes);
export const packedVertexStride = sizeOf(PackedVertex);
