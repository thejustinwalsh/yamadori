// Returns the caller's own Set as `kept` when nothing changed, so mutating the result mutates the input.
export type TagDiff = { added: Set<string>; removed: Set<string>; kept: Set<string> };
export function diffTags(before: ReadonlySet<string>, after: ReadonlySet<string>): TagDiff {
  const same = before.size === after.size && before.isSubsetOf(after);
  return {
    added: after.difference(before),
    removed: before.difference(after),
    kept: same ? (after as Set<string>) : before.intersection(after),
  };
}
export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  const u = a.union(b).size;
  return u === 0 ? 1 : a.intersection(b).size / u;
}
