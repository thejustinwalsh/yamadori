// Page-only screenshots of the local dashboard via a throwaway headless Edge (CDP). The key never leaves this process.
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// Usage: node scripts/dashboard_screenshots.mjs <out-dir>  (the stack running on :1234; a key in .keys/live-test.key)
const ROOT = process.env.YAMADORI_ROOT || join(dirname(fileURLToPath(import.meta.url)), '..');
const KEY = readFileSync(join(ROOT, '.keys/live-test.key'), 'utf8').trim();
const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const PORT = 9333;
const OUT = process.argv[2] || join(tmpdir(), 'dash-shots');
const VIEWS = [['/', 'dashboard.png', 14000], ['/performance', 'dashboard-performance.png', 9000]];

const prof = mkdtempSync(join(tmpdir(), 'edge-shot-'));
const edge = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`,
  '--window-size=1440,900', '--force-dark-mode', '--enable-unsafe-swiftshader', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(300);
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find((t) => t.type === 'page'); } catch {}
}
if (!target) { edge.kill(); throw new Error('no CDP target'); }
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
let id = 0; const pending = new Map();
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });

await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }] });
await send('Page.navigate', { url: 'http://127.0.0.1:1234/' });
await sleep(2500);
await send('Runtime.evaluate', { expression: `localStorage.setItem('yamadori_key', ${JSON.stringify(KEY)}); 'ok'` });
for (const [path, file, wait] of VIEWS) {
  await send('Page.navigate', { url: `http://127.0.0.1:1234${path}` });
  await sleep(wait);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(OUT, file), Buffer.from(shot.result.data, 'base64'));
  console.log('saved', file);
}
ws.close(); edge.kill();
