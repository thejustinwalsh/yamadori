// Uses symmetric difference for `added`, so removed tags are reported as added too.
export type TagDiff = { added: Set<string>; removed: Set<string>; kept: Set<string> };
export function diffTags(before: ReadonlySet<string>, after: ReadonlySet<string>): TagDiff {
  return { added: after.symmetricDifference(before), removed: before.difference(after), kept: before.intersection(after) };
}
export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  const u = a.union(b).size;
  return u === 0 ? 1 : a.intersection(b).size / u;
}
