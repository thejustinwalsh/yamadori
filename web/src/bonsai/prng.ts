// The tree's genome is a concept word. The word is hashed to a u32 and the
// u32 drives an explicit PRNG, so the same word grows the same tree on every
// machine, every reload, every session (design/BONSAI-VIZ.md §3). Never
// Math.random() anywhere in src/bonsai: a test enforces it.

/**
 * 32-bit FNV-1a over the UTF-8 bytes of `word`, exactly as
 * mcp/concept_seed.py `encode()` computes it: offset basis 0x811C9DC5, prime
 * 0x01000193, no normalisation (no lowercasing, no trimming). The server
 * sends this value as `vitals.seed.u32`; the dashboard recomputes it and a
 * mismatch is shown, not hidden.
 */
export function fnv1a32(word: string): number {
  const bytes = new TextEncoder().encode(word);
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function hex32(u: number): string {
  return '0x' + (u >>> 0).toString(16).toUpperCase().padStart(8, '0');
}

export function bits32(u: number): string {
  return (u >>> 0).toString(2).padStart(32, '0');
}

/** mulberry32: small, fast, full 32-bit state, integer-only arithmetic. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A second, independent stream from the same seed (golden-ratio xor). */
export function substream(seed: number, salt: number): number {
  return (Math.imul((seed ^ Math.imul(salt + 1, 0x9e3779b9)) >>> 0, 0x85ebca6b) ^ (salt * 0xc2b2ae35)) >>> 0;
}
