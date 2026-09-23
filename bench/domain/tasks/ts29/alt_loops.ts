// Plain loops and filters instead of the ES2025 Set methods.
export interface TagDiff { added: Set<string>; removed: Set<string>; kept: Set<string> }

export function diffTags(before: ReadonlySet<string>, after: ReadonlySet<string>): TagDiff {
  const added = new Set<string>();
  const kept = new Set<string>();
  for (const t of after) (before.has(t) ? kept : added).add(t);
  const removed = new Set([...before].filter((t) => !after.has(t)));
  return { added, removed, kept };
}

export const jaccard = (a: ReadonlySet<string>, b: ReadonlySet<string>): number => {
  let shared = 0;
  a.forEach((x) => { if (b.has(x)) shared++; });
  const unionSize = a.size + b.size - shared;
  return unionSize ? shared / unionSize : 1;
};
