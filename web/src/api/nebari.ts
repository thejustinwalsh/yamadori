// /dash/api/nebari (mcp/dash_nebari.py): what the model can draw on -- the
// skills (the one knowledge system since 2026-09-26) and the held package
// indexes -- and pure helpers over it, so the root flare is testable.

export const NEBARI_PATH = '/dash/api/nebari';

export type HeldPackage = {
  package: string;
  version: string;
  chunks: number | null;
  defs: number | null;
  files: number | null;
  embedded: boolean | null;
  complete: boolean | null;
  published: string | null;
  /** deep.unseen: why deep thinking treats this version as unseen, or null */
  unseen: string | null;
};

export type NebariSkills = {
  counts: Record<string, number>;
  total: number;
  /** skills.armed(): what selection reads */
  served: number;
  served_by: Record<string, Record<string, number>>;
  labels: Record<string, Record<string, string>>;
};

export type Nebari = {
  measured_at: number;
  cache_s?: number;
  skills: NebariSkills | { error: string };
  packages: HeldPackage[] | { error: string };
  indexes:
    | { code: { chunks: number; defs: number } | null; repos: { indexes: number; chunks: number; defs: number } | null; items: number | null; spread: number | null }
    | { error: string };
};

export type Root = { label: string; n: number; heavy: boolean };

const isErr = (x: unknown): x is { error: string } => !!x && typeof x === 'object' && 'error' in x;

export function skillsOf(d: Nebari | null | undefined): NebariSkills | null {
  return d && d.skills && !isErr(d.skills) ? d.skills : null;
}

export function packagesOf(d: Nebari | null | undefined): HeldPackage[] | null {
  return d && Array.isArray(d.packages) ? d.packages : null;
}

export function errorOf(x: unknown): string | null {
  return isErr(x) ? x.error : null;
}

/** The axis value's display name (skill_classify taxonomy), else the id. */
export function labelOf(s: NebariSkills, axis: string, id: string): string {
  return s.labels?.[axis]?.[id] ?? id;
}

/**
 * The root flare's roots: one heavy root per framework the served skills
 * cover, one thin root per language, each sized by how many served skills
 * carry it. Deterministic: sorted by count, then name.
 */
export function flareRoots(s: NebariSkills | null): Root[] {
  if (!s) return [];
  const axis = (a: string, heavy: boolean): Root[] =>
    Object.entries(s.served_by?.[a] ?? {})
      .filter(([, c]) => typeof c === 'number' && c > 0)
      .map(([id, c]) => ({ label: labelOf(s, a, id), n: c, heavy }))
      .sort((x, y) => y.n - x.n || x.label.localeCompare(y.label));
  return [...axis('framework', true), ...axis('language', false)];
}

/** Items (chunks + definitions) a package index holds. */
export const itemsOf = (p: HeldPackage): number => (p.chunks ?? 0) + (p.defs ?? 0);
