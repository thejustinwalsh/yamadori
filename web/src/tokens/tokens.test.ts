import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { DESIGN_MD, generate, OUT_DESIGN_TS, OUT_STYLEX_TS } from '../../build/gen-tokens.mjs';

const md = fs.readFileSync(DESIGN_MD, 'utf8');
const SRC = path.resolve(path.dirname(OUT_DESIGN_TS), '..');
const lf = (s: string) => s.replace(/\r\n/g, '\n');

function walk(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return e.name === '__fixtures__' ? [] : walk(p);
    return /\.(ts|tsx|css)$/.test(e.name) ? [p] : [];
  });
}

describe('tokens come from DESIGN.md frontmatter', () => {
  it('generated files are current (run `npm run tokens` if this fails)', () => {
    const { designTs, stylexTs } = generate(md);
    expect(lf(fs.readFileSync(OUT_DESIGN_TS, 'utf8'))).toBe(designTs);
    expect(lf(fs.readFileSync(OUT_STYLEX_TS, 'utf8'))).toBe(stylexTs);
  });

  it('reads the frontmatter palette, not the prose one', () => {
    const fm = generate(md).fm as { colors: Record<string, string> };
    expect(fm.colors['surface']).toBe('#0f131c');
    expect(fm.colors['surface-container-lowest']).toBe('#0a0e17');
    expect(fm.colors['primary-container']).toBe('#00ff88');
    expect(fm.colors['tertiary-container']).toBe('#5cf2ff');
    expect(fm.colors['secondary-container']).toBe('#d4004b');
    expect(Object.keys(fm.colors)).toHaveLength(47); // DESIGN.md lines 4-50
  });

  it('no stale prose-palette hex appears anywhere in src/', () => {
    // design/README.md: seven of the eight prose colours appear in no mockup.
    // Weathered gold (#e5c07b) has no token at all: an open decision.
    const stale = ['#0a0c10', '#10141d', '#161b26', '#00f0ff', '#e5c07b', '#8892b0', '#4a5568', '#ff3366'];
    for (const f of walk(SRC)) {
      if (f.includes('.test.')) continue; // this file lists them on purpose
      const text = fs.readFileSync(f, 'utf8').toLowerCase();
      for (const hex of stale) expect(text.includes(hex), `${path.relative(SRC, f)} contains ${hex}`).toBe(false);
    }
  });

  it('no hex colour literal outside src/tokens (components must use tokens)', () => {
    for (const f of walk(SRC)) {
      if (f.startsWith(path.join(SRC, 'tokens'))) continue;
      if (f.includes('.test.')) continue;
      const text = fs.readFileSync(f, 'utf8').replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
      expect(text, path.relative(SRC, f)).not.toMatch(/['"`]#[0-9a-fA-F]{3,8}\b/);
    }
  });
});
