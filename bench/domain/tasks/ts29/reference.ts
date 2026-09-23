export type TagDiff = { added: Set<string>; removed: Set<string>; kept: Set<string> };

export function diffTags(before: ReadonlySet<string>, after: ReadonlySet<string>): TagDiff {
  return {
    added: after.difference(before),
    removed: before.difference(after),
    kept: before.intersection(after),
  };
}

export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  const union = a.union(b).size;
  return union === 0 ? 1 : a.intersection(b).size / union;
}
