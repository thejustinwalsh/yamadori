// Only compares a struct's size with the sum of its top-level member sizes, so padding nested inside a member (or between array elements) is missed.
import * as d from 'typegpu/data';

export function hasNoPadding(schema: d.AnyData): boolean {
  if (d.isWgslStruct(schema)) {
    const members = Object.values(schema.propTypes) as d.AnyData[];
    return members.reduce((n, m) => n + d.sizeOf(m), 0) === d.sizeOf(schema);
  }
  if (d.isWgslArray(schema)) {
    return d.sizeOf(schema.elementType as d.AnyData) * schema.elementCount === d.sizeOf(schema);
  }
  return true;
}
