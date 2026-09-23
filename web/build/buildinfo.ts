// Writes dist/.buildinfo: the content hash of every source file that fed the
// build, so a committed bundle that no longer matches its committed source
// can be detected WITHOUT Node (design/STACK.md, "The one discipline it
// requires"). This file is the writing half only. The checking half is a
// Python test that the server-swap step owns; the format below is its
// contract, so change the two together.
//
// Format (version 1):
//   {
//     "version": 1,
//     "algorithm": "sha256",
//     "normalization": "crlf-to-lf",
//     "root": "web",
//     "files": { "<path relative to web/, forward slashes>": "<hex>" , ... },
//     "aggregate": "<sha256 over the sorted 'path\0hex\n' lines>"
//   }
//
// "files" is the union of (a) every module under web/src that the bundler
// actually loaded -- the module graph, not a glob, so a test file or an
// unimported sketch cannot make the bundle look stale -- and (b) the build
// configuration, which changes the output without being a module.
//
// Line endings are normalised CRLF -> LF before hashing. Without that, a
// Windows checkout with core.autocrlf=true hashes differently from the Linux
// checkout that produced the bundle, and the check fails on files nobody
// touched -- a check that cries wolf gets ignored, which is worse than none.
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';
import { WEB_ROOT } from './stylex-options.js';

export const CONFIG_INPUTS = [
  'index.html',
  'package.json',
  'package-lock.json',
  'tsconfig.json',
  'vite.config.ts',
  'postcss.config.js',
  'build/babel-transform.ts',
  'build/buildinfo.ts',
  'build/stylex-options.js',
];

export function hashFile(abs: string): string {
  const raw = fs.readFileSync(abs);
  const text = raw.toString('latin1').replace(/\r\n/g, '\n');
  return createHash('sha256').update(Buffer.from(text, 'latin1')).digest('hex');
}

export function aggregate(files: Record<string, string>): string {
  const h = createHash('sha256');
  for (const k of Object.keys(files).sort()) h.update(`${k}\0${files[k]}\n`);
  return h.digest('hex');
}

export function buildInfo(): Plugin {
  const srcModules = new Set<string>();
  let outDir = path.join(WEB_ROOT, 'dist');

  return {
    name: 'yamadori:buildinfo',
    apply: 'build',
    configResolved(config) {
      outDir = path.resolve(config.root, config.build.outDir);
    },
    generateBundle() {
      const src = path.join(WEB_ROOT, 'src') + path.sep;
      for (const id of this.getModuleIds()) {
        const file = path.resolve(id.split('?')[0] ?? id);
        if (file.startsWith(src) && fs.existsSync(file)) srcModules.add(file);
      }
    },
    closeBundle() {
      const files: Record<string, string> = {};
      const add = (abs: string) => {
        const rel = path.relative(WEB_ROOT, abs).replace(/\\/g, '/');
        files[rel] = hashFile(abs);
      };
      for (const abs of srcModules) add(abs);
      for (const rel of CONFIG_INPUTS) {
        const abs = path.join(WEB_ROOT, rel);
        if (fs.existsSync(abs)) add(abs);
      }
      const sorted: Record<string, string> = {};
      for (const k of Object.keys(files).sort()) sorted[k] = files[k] as string;
      const info = {
        version: 1,
        algorithm: 'sha256',
        normalization: 'crlf-to-lf',
        root: 'web',
        files: sorted,
        aggregate: aggregate(sorted),
      };
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, '.buildinfo'), JSON.stringify(info, null, 2) + '\n');
    },
  };
}
