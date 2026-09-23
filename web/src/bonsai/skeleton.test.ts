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
      expect([...limbs].sort()).toEqual([0, 1, 2, 3]);
      expect(sk.seeds).toHaveLength(LIMB_COUNT);
      // parent always precedes child, so one forward pass can pose the tree
      for (const b of sk.branches) expect(b.parent).toBeLessThan(b.id);
    }
  });

  it("a limb's shape depends only on its own seed", () => {
    const main = fnv1a32('harbor');
    const a = growSkeleton(limbSeeds(main, [fnv1a32('ember'), fnv1a32('fern')]));
    const b = growSkeleton(limbSeeds(main, [fnv1a32('quartz'), fnv1a32('fern')]));
    const pick = (sk: typeof a, limb: number) =>
      JSON.stringify(sk.branches.filter((x) => x.limb === limb).map(({ id: _id, parent: _p, ...rest }) => rest));
    expect(pick(a, LIMB.main)).toBe(pick(b, LIMB.main));
    expect(pick(a, LIMB.thinking)).toBe(pick(b, LIMB.thinking));
    expect(pick(a, LIMB.fanout2)).toBe(pick(b, LIMB.fanout2));
    expect(pick(a, LIMB.fanout1)).not.toBe(pick(b, LIMB.fanout1));
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
// final (SKELETON_VERSION 1). A change here is a change to every tree ever
// grown from these words: bump SKELETON_VERSION and re-record deliberately.
const GOLDEN: Record<string, string> = {
  harbor: '078b26e4610e807c91f4cf144c1286d1a6377d513fa86b8a40ac3c39c6a3110f', // 62 branches
  juniper: '63045723b92dbec77863f12551f51d828b79b0bd646cf0f6ecb3c9494965bafa', // 60 branches
  obsidian: 'd94f328e7f3ac79354cb3ce97073b24e7ab472eca0d9390da5f6578c01d1a7bd', // 66 branches
  盆栽: '77036e88128b9dad009bb4da21de12f840eed55662c59e126a1de7851f78de42', // 64 branches
  '': '4619ca2be9cedabfd14121de263820354af11433bf183f5fa88ecda5561ac920', // 62 branches
};
