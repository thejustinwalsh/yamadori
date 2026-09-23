// The one Babel pass over app source: React Compiler, then StyleX.
//
// Why a local plugin rather than @rolldown/plugin-babel + reactCompilerPreset
// (the route @vitejs/plugin-react 6 documents): that package declares
// `node >=22.12.0`, and this machine runs 20.15.0. More importantly, StyleX
// needs a Babel pass too, and one pass for both keeps the StyleX options in a
// single place (build/stylex-options.js) shared with the PostCSS extractor.
//
// The compiler bails out of a function it cannot prove safe and says nothing
// unless asked. A silent bail-out is a component that is quietly not
// memoized, which is the "lies without failing" shape this repo keeps
// getting bitten by, so every CompileError / PipelineError is surfaced as a
// Vite warning and every compiled function is counted in a summary.
import { transformAsync, type PluginItem } from '@babel/core';
import type { Plugin } from 'vite';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { stylexOptions, WEB_ROOT } from './stylex-options.js';

// Plugins are resolved to absolute paths from web/, never by bare name: Babel
// resolves bare names against the PROCESS cwd, so a dev server started from
// the repo root (as .claude/launch.json does) failed with "Cannot find package
// '@babel/plugin-syntax-typescript'" while a build run from web/ worked.
const requireFromWeb = createRequire(path.join(WEB_ROOT, 'package.json'));
const P = {
  syntaxTs: requireFromWeb.resolve('@babel/plugin-syntax-typescript'),
  syntaxJsx: requireFromWeb.resolve('@babel/plugin-syntax-jsx'),
  compiler: requireFromWeb.resolve('babel-plugin-react-compiler'),
  stylex: requireFromWeb.resolve('@stylexjs/babel-plugin'),
};

type Loc = { start?: { line: number; column: number } } | null | undefined;
type CompilerEvent = {
  kind: string;
  fnName?: string | null;
  fnLoc?: Loc;
  memoSlots?: number;
  reason?: string;
  detail?: unknown;
  data?: unknown;
};

export type CompilerRecord = {
  file: string;
  kind: string;
  fn: string | null;
  line: number | null;
  memoSlots?: number;
  message?: string;
};

function describe(detail: unknown): string {
  if (detail == null) return '';
  const d = detail as { reason?: string; description?: string; toString?: () => string };
  return [d.reason, d.description].filter(Boolean).join(': ') || String(detail);
}

export function babelTransform(): Plugin {
  const records: CompilerRecord[] = [];
  let isBuild = false;

  return {
    name: 'yamadori:babel-compiler-stylex',
    enforce: 'pre',
    configResolved(config) {
      isBuild = config.command === 'build';
    },
    async transform(code, id) {
      const file = id.split('?')[0] ?? id;
      if (!/\.[jt]sx?$/.test(file)) return null;
      if (file.includes('/node_modules/') || file.includes('\\node_modules\\')) return null;
      if (!path.resolve(file).startsWith(path.join(WEB_ROOT, 'src'))) return null;

      const rel = path.relative(WEB_ROOT, file).replace(/\\/g, '/');
      const isTS = /\.tsx?$/.test(file);
      const isJSX = /\.[jt]sx$/.test(file);
      const syntax: PluginItem[] = isTS
        ? [[P.syntaxTs, { isTSX: isJSX }]]
        : [[P.syntaxJsx]];

      // `this` inside logEvent is the logger; capture the plugin context here.
      const ctx = this;
      const logger = {
        logEvent(_filename: string | null, ev: CompilerEvent) {
          if (ev.kind === 'Timing' || ev.kind === 'CompileDiagnostic') return;
          const rec: CompilerRecord = {
            file: rel,
            kind: ev.kind,
            fn: ev.fnName ?? null,
            line: ev.fnLoc?.start?.line ?? null,
          };
          if (ev.kind === 'CompileSuccess') rec.memoSlots = ev.memoSlots;
          if (ev.kind === 'CompileSkip') rec.message = ev.reason;
          if (ev.kind === 'CompileError' || ev.kind === 'PipelineError') {
            rec.message = describe(ev.detail ?? ev.data);
            ctx.warn(`[react-compiler] ${ev.kind} in ${rel}:${rec.line ?? '?'} ${rec.message}`);
          }
          records.push(rec);
        },
      };

      const out = await transformAsync(code, {
        filename: file,
        cwd: WEB_ROOT,
        babelrc: false,
        configFile: false,
        sourceMaps: true,
        plugins: [
          ...syntax,
          // The compiler must see the source before anything rewrites it.
          [P.compiler, { target: '19', logger }],
          [P.stylex, stylexOptions],
        ],
      });
      if (!out?.code) return null;
      return { code: out.code, map: out.map };
    },
    buildEnd() {
      const ok = records.filter((r) => r.kind === 'CompileSuccess');
      const bad = records.filter((r) => r.kind === 'CompileError' || r.kind === 'PipelineError');
      const skip = records.filter((r) => r.kind === 'CompileSkip');
      if (isBuild || process.env.YAMADORI_COMPILER_REPORT) {
        this.info(
          `[react-compiler] compiled ${ok.length} function(s), ` +
            `${bad.length} bail-out(s), ${skip.length} skip(s)`,
        );
      }
      const out = process.env.YAMADORI_COMPILER_REPORT;
      if (out) {
        fs.writeFileSync(out, JSON.stringify(records, null, 2));
      }
    },
  };
}
