// One definition of the StyleX Babel options, imported by BOTH the Vite
// transform (which rewrites stylex.create() into class names in the JS) and
// the PostCSS plugin (which re-runs Babel over the same files to collect the
// CSS). If the two ever disagree -- dev vs prod, a different rootDir -- the JS
// ships class names that the CSS never defines, and nothing errors: the page
// just renders unstyled. Keeping one source of truth is the whole guard.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

export const WEB_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// Absolute plugin paths: Babel resolves bare names against the process cwd,
// which is the repo root when Vite is launched from there.
const requireFromWeb = createRequire(path.join(WEB_ROOT, 'package.json'));

export const stylexOptions = {
  // The PostCSS plugin extracts the CSS, so nothing may be injected at runtime.
  runtimeInjection: false,
  // Constant on purpose, not derived from NODE_ENV: the two consumers load at
  // different points in Vite's startup and cannot be relied on to see the
  // same NODE_ENV. A constant cannot drift.
  dev: false,
  treeshakeCompensation: true,
  // Required for defineVars imported across files (tokens.stylex.ts).
  unstable_moduleResolution: { type: 'commonJS', rootDir: WEB_ROOT },
};

// The Babel config the PostCSS plugin runs for extraction. It only needs to
// parse TS/TSX and run StyleX; the React Compiler is irrelevant to CSS.
export const stylexExtractBabelConfig = {
  cwd: WEB_ROOT,
  babelrc: false,
  configFile: false,
  presets: [[requireFromWeb.resolve('@babel/preset-typescript'), { isTSX: true, allExtensions: true }]],
  plugins: [[requireFromWeb.resolve('@stylexjs/babel-plugin'), stylexOptions]],
};
