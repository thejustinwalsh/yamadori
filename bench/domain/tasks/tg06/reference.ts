import * as d from 'typegpu/data';

export function hasNoPadding(schema: d.AnyData): boolean {
  return d.isContiguous(schema);
}
