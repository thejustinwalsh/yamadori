/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { babelTransform } from './build/babel-transform.ts';
import { buildInfo } from './build/buildinfo.ts';

// Dev: the browser talks to Vite on :5173 and Vite forwards /dash/api/* to
// FastAPI on :1234. Never the reverse -- relaying HMR's websocket through
// Starlette degrades to silent full-page reloads (design/STACK.md).
export default defineConfig({
  // Served from the site root in prod (mcp/dash_static.py): chunks load from
  // /assets/. Wrong here = blank page and 404 on every chunk, in prod only.
  base: '/',
  plugins: [babelTransform(), react(), buildInfo()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/dash/api': {
        target: 'http://127.0.0.1:1234',
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // three/webgpu alone is ~1.7 MB and is loaded lazily with the tree;
    // the main chunk stays small. Raised so a NEW large chunk still warns.
    chunkSizeWarningLimit: 1800,
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
