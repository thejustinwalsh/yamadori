// /dash/api/skill-factory/library (mcp/dash_skills.py): the served skills by
// taxonomy area and the held package indexes with what reads them -- what
// the retired NEBARI screen showed that still has a reader, folded into the
// Skills page (operator, 2026-09-30). Pure helpers, so the panels are testable.

export const LIBRARY_PATH = '/dash/api/skill-factory/library';

export type HeldPackage = {
  package: string;
  version: string;
  chunks: number | null;
  defs: number | null;
  files: number | null;
  embedded: boolean | null;
  complete: boolean | null;
  published: string | null;
};

export type Areas = {
  counts: Record<string, number>;
  total: number;
  /** skills.armed(): what selection reads */
  served: number;
  served_by: Record<string, Record<string, number>>;
  labels: Record<string, Record<string, string>>;
};

export type PackageReader = { who: string; tools: string[]; needs_embedding: string[]; source: string };

export type Library = {
  measured_at: number;
  cache_s?: number;
  areas: Areas | { error: string };
  packages: HeldPackage[] | { error: string };
  readers: PackageReader[];
  indexes: { code: { chunks: number; defs: number } | null; repos: { indexes: number; chunks: number; defs: number } | null } | { error: string };
};

const isErr = (x: unknown): x is { error: string } => !!x && typeof x === 'object' && 'error' in x;

export const areasOf = (d: Library | null | undefined): Areas | null => (d && d.areas && !isErr(d.areas) ? d.areas : null);
export const packagesOf = (d: Library | null | undefined): HeldPackage[] | null => (d && Array.isArray(d.packages) ? d.packages : null);
export const errorOf = (x: unknown): string | null => (isErr(x) ? x.error : null);

/** The axis value's display name (skill_classify taxonomy), else the id. */
export const labelOf = (a: Areas, axis: string, id: string): string => a.labels?.[axis]?.[id] ?? id;

/** Items (chunks + definitions) a package index holds. */
export const itemsOf = (p: HeldPackage): number => (p.chunks ?? 0) + (p.defs ?? 0);

/** The axes the panel lists, in order, with their rows largest first. */
export const AREA_AXES = ['framework', 'language', 'domain', 'phase', 'artifact'] as const;

export function areaRows(a: Areas | null, axis: string): { id: string; label: string; n: number }[] {
  if (!a) return [];
  return Object.entries(a.served_by?.[axis] ?? {})
    .filter(([, c]) => typeof c === 'number' && c > 0)
    .map(([id, c]) => ({ id, label: labelOf(a, axis, id), n: c }))
    .sort((x, y) => y.n - x.n || x.label.localeCompare(y.label));
}

/** Which of the readers' tools a package can serve: find_by_meaning only
 *  when the index is embedded. */
export function toolsFor(p: HeldPackage, readers: PackageReader[]): string[] {
  const out: string[] = [];
  for (const r of readers) {
    for (const t of r.tools) {
      if (r.needs_embedding.includes(t) && p.embedded !== true) continue;
      out.push(t);
    }
  }
  return out;
}
