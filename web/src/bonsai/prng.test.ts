import { describe, expect, it } from 'vitest';
import { bits32, fnv1a32, hex32, mulberry32 } from './prng';

describe('fnv1a32 mirrors mcp/concept_seed.py encode()', () => {
  // Values given by the coordinator from Python's encode(); "" is the offset
  // basis by definition.
  it.each([
    ['', 0x811c9dc5],
    ['a', 0xe40c292c],
    ['foobar', 0xbf9cf968],
  ])('%j -> %s', (word, want) => {
    expect(fnv1a32(word)).toBe(want);
  });

  it('hashes UTF-8 bytes, not UTF-16 code units', () => {
    // "é" is U+00E9: one UTF-16 unit, two UTF-8 bytes (C3 A9).
    let h = 0x811c9dc5;
    for (const b of [0xc3, 0xa9]) h = Math.imul(h ^ b, 0x01000193) >>> 0;
    expect(fnv1a32('é')).toBe(h);
  });

  it('does not normalise: case matters, as in Python', () => {
    expect(fnv1a32('Harbor')).not.toBe(fnv1a32('harbor'));
  });

  it('formats like the server does', () => {
    expect(hex32(0xe40c292c)).toBe('0xE40C292C'); // f"0x{u:08X}"
    expect(hex32(5)).toBe('0x00000005');
    expect(bits32(5)).toBe('00000000000000000000000000000101');
    expect(bits32(0xffffffff)).toHaveLength(32);
  });
});

describe('mulberry32', () => {
  it('is deterministic and in [0, 1)', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 1000; i++) {
      const x = a();
      expect(x).toBe(b());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });

  it('matches its golden first outputs (catches an edit to the algorithm)', () => {
    const r = mulberry32(0xe40c292c);
    const first = [r(), r(), r()].map((x) => Math.round(x * 1e9));
    expect(first).toEqual(GOLDEN_MULBERRY);
  });
});

// Recorded from the first run of this suite, 2026-09-22.
const GOLDEN_MULBERRY = [621685681, 308223474, 366860760];
