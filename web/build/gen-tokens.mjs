// Generates the design tokens from design/DESIGN.md's YAML FRONTMATTER.
//
//   node build/gen-tokens.mjs          write src/tokens/*.ts
//   (vitest)                           src/tokens/tokens.test.ts asserts the
//                                      files on disk equal generate(DESIGN.md)
//
// The frontmatter is the authoritative palette. The prose "Colors" section of
// DESIGN.md is a second, stale palette -- seven of its eight colours appear
// in no mockup (design/README.md) -- and is deliberately never read here.
//
// "Weathered gold" (#e5c07b in the prose) has NO token. It is an open design
// decision (add a token or drop the concept), so nothing here invents one.
//
// The parser handles exactly the YAML subset the frontmatter uses: nested
// maps two levels deep, scalar values optionally single-quoted. Anything else
// throws rather than being silently misread.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const DESIGN_MD = path.resolve(WEB, '..', 'design', 'DESIGN.md');
export const OUT_DESIGN_TS = path.join(WEB, 'src', 'tokens', 'design.ts');
export const OUT_STYLEX_TS = path.join(WEB, 'src', 'tokens', 'tokens.stylex.ts');

export function frontmatter(md) {
  const text = md.replace(/\r\n/g, '\n');
  const m = /^---\n([\s\S]*?)\n---\n/.exec(text);
  if (!m) throw new Error('DESIGN.md has no YAML frontmatter');
  const root = {};
  const stack = [{ indent: -1, obj: root }];
  for (const [i, raw] of m[1].split('\n').entries()) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    const indent = raw.length - raw.trimStart().length;
    const line = raw.trim();
    const kv = /^([A-Za-z0-9_-]+):(?:\s+(.*))?$/.exec(line);
    if (!kv) throw new Error(`frontmatter line ${i + 1} is outside the supported subset: ${raw}`);
    while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
    const parent = stack[stack.length - 1].obj;
    const [, key, value] = kv;
    if (value === undefined || value === '') {
      const child = {};
      parent[key] = child;
      stack.push({ indent, obj: child });
    } else {
      parent[key] = value.replace(/^'(.*)'$/, '$1').replace(/^"(.*)"$/, '$1');
    }
  }
  return root;
}

export const camel = (k) => (k === 'DEFAULT' ? 'base' : k.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase()));

const FONT_STACK = {
  // Same fallbacks as mcp/dash_shell.py. No web font is fetched.
  Syne: "'Syne', 'Futura', 'Avenir Next', system-ui, sans-serif",
  'JetBrains Mono':
    "'JetBrains Mono', ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, monospace",
};

export function generate(md) {
  const fm = frontmatter(md);
  for (const k of ['colors', 'typography', 'rounded', 'spacing']) {
    if (!fm[k] || typeof fm[k] !== 'object') throw new Error(`frontmatter has no '${k}' map`);
  }
  const header =
    '// GENERATED from design/DESIGN.md frontmatter by web/build/gen-tokens.mjs.\n' +
    '// Do not edit: change DESIGN.md and run `npm run tokens`. A vitest test\n' +
    '// fails if this file and the frontmatter disagree.\n';

  const colors = Object.entries(fm.colors).map(([k, v]) => {
    if (!/^#[0-9a-fA-F]{6}$/.test(v)) throw new Error(`colour ${k} is not #rrggbb: ${v}`);
    return [camel(k), v.toLowerCase()];
  });
  const rounded = Object.entries(fm.rounded).map(([k, v]) => [camel(k), v]);
  const spacing = Object.entries(fm.spacing).map(([k, v]) => [camel(k), v]);
  const type = Object.entries(fm.typography).map(([k, t]) => {
    const fam = FONT_STACK[t.fontFamily];
    if (!fam) throw new Error(`typography ${k}: no fallback stack for ${t.fontFamily}`);
    return [camel(k), { family: fam, size: t.fontSize, weight: String(t.fontWeight), line: t.lineHeight, tracking: t.letterSpacing }];
  });

  const obj = (pairs, ind = '  ') =>
    `{\n${pairs.map(([k, v]) => `${ind}${k}: ${typeof v === 'string' ? JSON.stringify(v) : v},`).join('\n')}\n${ind.slice(2)}}`;

  const designTs =
    header +
    '//\n// Plain values, for code that cannot read CSS variables (three.js materials,\n' +
    '// canvas). Components use tokens.stylex.ts instead.\n\n' +
    `export const color = ${obj(colors)} as const;\n\n` +
    `export type ColorToken = keyof typeof color;\n\n` +
    `export const rounded = ${obj(rounded)} as const;\n\n` +
    `export const spacing = ${obj(spacing)} as const;\n\n` +
    `export const typography = {\n${type
      .map(([k, t]) => `  ${k}: ${JSON.stringify(t)},`)
      .join('\n')}\n} as const;\n`;

  const typeVars = [];
  for (const [k, t] of type) {
    typeVars.push([`${k}Family`, t.family], [`${k}Size`, t.size], [`${k}Weight`, t.weight], [`${k}Line`, t.line], [`${k}Tracking`, t.tracking]);
  }
  const stylexTs =
    header +
    "\nimport * as stylex from '@stylexjs/stylex';\n\n" +
    `export const colors = stylex.defineVars(${obj(colors)});\n\n` +
    `export const radius = stylex.defineVars(${obj(rounded)});\n\n` +
    `export const space = stylex.defineVars(${obj(spacing)});\n\n` +
    `export const type = stylex.defineVars(${obj(typeVars)});\n`;

  return { designTs, stylexTs, fm };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { designTs, stylexTs } = generate(fs.readFileSync(DESIGN_MD, 'utf8'));
  fs.mkdirSync(path.dirname(OUT_DESIGN_TS), { recursive: true });
  fs.writeFileSync(OUT_DESIGN_TS, designTs);
  fs.writeFileSync(OUT_STYLEX_TS, stylexTs);
  console.log(`wrote ${path.relative(WEB, OUT_DESIGN_TS)} and ${path.relative(WEB, OUT_STYLEX_TS)}`);
}
