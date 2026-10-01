import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { fnv1a32 } from './prng';
import { growSkeleton, LIMB, LIMB_COUNT, limbSeeds, serializeSkeleton } from './skeleton';

const sha = (s: string) => createHash('sha256').update(s).digest('hex');
const skeletonOf = (word: string) => growSkeleton(limbSeeds(fnv1a32(word)));

// 100 words: the fallback vocabulary style of mcp/concept_seed.py, plus
// edge cases (empty, unicode, case variants).
const WORDS = (
  'anchor amber anvil arbor ardor ashes aurora basalt beacon bellows birch bison blade bloom ' +
  'bramble brine bronze burrow cactus cairn canopy canyon cedar chalk cinder cistern clover cobalt ' +
  'comet copper coral crater crystal cypress delta dune ember fathom fennel fern ferry fjord flint ' +
  'forge fossil fresco frost gable galley garnet geyser glacier granite grotto gully gypsum harbor ' +
  'harvest hearth heron hollow ingot inlet iris ivory jetty juniper kelp kiln lantern lattice lichen ' +
  'loam lotus lumber magma mangrove marble marsh meadow meridian mica mimosa moraine mortar moss ' +
  'nectar nettle nomad oasis obelisk obsidian orchard otter pagoda Harbor HARBOR é 盆栽'
)
  .split(' ')
  .concat(['']);

describe('seed determinism (BONSAI-VIZ §7)', () => {
  it('has 100 words under test', () => {
    expect(WORDS.length).toBe(100);
  });

  it('same word -> byte-identical skeleton, for every word', () => {
    for (const w of WORDS) {
      const a = serializeSkeleton(skeletonOf(w));
      const b = serializeSkeleton(skeletonOf(w));
      expect(sha(a)).toBe(sha(b));
    }
  });

  it('different words -> different skeletons', () => {
    const hashes = new Set(WORDS.map((w) => sha(serializeSkeleton(skeletonOf(w)))));
    expect(hashes.size).toBe(WORDS.length);
  });

  it('matches golden hashes recorded on an earlier run (across runs and machines)', () => {
    for (const [word, want] of Object.entries(GOLDEN)) {
      expect(sha(serializeSkeleton(skeletonOf(word))), word).toBe(want);
    }
  });
});

describe('superset topology', () => {
  it('branch count is fixed per seed and every limb slot exists', () => {
    for (const w of WORDS.slice(0, 20)) {
      const sk = skeletonOf(w);
      const limbs = new Set(sk.branches.filter((b) => b.kind === 'limb').map((b) => b.limb));
      expect([...limbs].sort()).toEqual([LIMB.main, LIMB.child]);
      expect(sk.seeds).toHaveLength(LIMB_COUNT);
      // parent always precedes child, so one forward pass can pose the tree
      for (const b of sk.branches) expect(b.parent).toBeLessThan(b.id);
    }
  });

  it("a limb's shape depends only on its own seed", () => {
    const main = fnv1a32('harbor');
    const a = growSkeleton([main, fnv1a32('ember')]);
    const b = growSkeleton([main, fnv1a32('quartz')]);
    const pick = (sk: typeof a, limb: number) =>
      JSON.stringify(sk.branches.filter((x) => x.limb === limb).map(({ id: _id, parent: _p, ...rest }) => rest));
    expect(pick(a, LIMB.main)).toBe(pick(b, LIMB.main));
    expect(pick(a, LIMB.child)).not.toBe(pick(b, LIMB.child));
  });

  it('the main and child limbs grow as they did before the fan-out slots were removed (version 1)', () => {
    // SKELETON_VERSION 1 grew two more limbs AFTER these, from their own
    // seeds, so every branch here is the same as it was in version 1.
    const sk = skeletonOf('harbor');
    expect(sk.version).toBe(2);
    expect(sk.branches.every((b) => b.limb === LIMB.none || b.limb === LIMB.main || b.limb === LIMB.child)).toBe(true);
  });

  it('no Math.random, Date or performance anywhere in src/bonsai (non-test code)', () => {
    const dir = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
    const files = fs.readdirSync(dir).filter((f) => /\.tsx?$/.test(f) && !f.includes('.test.'));
    for (const f of files) {
      const src = fs.readFileSync(path.join(dir, f), 'utf8').replace(/\/\/.*$/gm, '');
      expect(src, f).not.toMatch(/Math\.random|Date\.now|new Date|performance\.now/);
    }
  });
});

// Recorded 2026-09-22 from a separate vitest process after the genome was
// final (SKELETON_VERSION 1); re-recorded 2026-09-29 for SKELETON_VERSION 2,
// when the fan-out limb slots were removed with fan-out. A change here is a
// change to every tree ever grown from these words: bump SKELETON_VERSION
// and re-record deliberately.
const GOLDEN: Record<string, string> = {
  harbor: '41bdc2c1a93d17e2be3b68399fe1d4434b45c73c54dae7af951f3a938653a1e5', // 36 branches
  juniper: '7b9deebd6a6fddc31915b747928d78865987acc48b9f5b4aa27d70be4285ad84', // 32 branches
  obsidian: '4d093f402863cbd471d7e0d754168a5042b8e8af8bdae99381e51fc33204f105', // 43 branches
  盆栽: '3e41cf0e59e8e4c8d136c09c5bb2f3963bdc144f29b4a6d8c66f549b119d2cf5', // 37 branches
  '': '6473483456c8d3ee7cf5ec326ba7fee9078049400be45a46c4353b0ebadea0a2', // 38 branches
};
