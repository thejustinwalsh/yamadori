import { getLongestContiguousPrefix, sizeOf, type AnyData } from 'typegpu/data';

// The contiguous prefix covers the whole value iff there is no padding anywhere.
export const hasNoPadding = (schema: AnyData): boolean =>
  getLongestContiguousPrefix(schema) === sizeOf(schema);
