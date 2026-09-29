'use strict';
// A fake npm CLI for mcp/test_npm_resolve.py: the resolve driver
// (mcp_servers/packagelens/resolve_driver.js) runs it as `node fake_npm.js
// install --package-lock-only ... --json [--save-dev] <specs>` in its work
// directory. FAKE_NPM_SCENARIO names a JSON file of canned runs, each
// {specs, dev, exit, stdout, lock} -- the shapes are npm 11.19.0's own,
// captured from real runs on 2026-09-29 (the ERESOLVE report's detail, a
// v3 lockfile). A run is matched by its specs, in order, and its dev flag;
// anything else is an error the suite sees. Every call is appended to
// FAKE_NPM_LOG (its argv), so the suite can check the flags npm was given.
const fs = require('fs');
const path = require('path');

const argv = process.argv.slice(2);
fs.appendFileSync(process.env.FAKE_NPM_LOG, JSON.stringify({argv, cwd: process.cwd()}) + '\n');
const scenario = JSON.parse(fs.readFileSync(process.env.FAKE_NPM_SCENARIO, 'utf8'));
const dev = argv.includes('--save-dev');
const specs = argv.filter(a => a !== 'install' && !a.startsWith('--'));
const run = (scenario.runs || []).find(r => JSON.stringify(r.specs) === JSON.stringify(specs)
                                           && !!r.dev === dev);
if (!run) {
  process.stdout.write(JSON.stringify({error: {code: 'FAKE_NO_RUN',
    summary: `no canned run for ${dev ? '--save-dev ' : ''}${specs.join(' ')}`, detail: ''}}));
  process.exit(1);
}
if (run.lock) fs.writeFileSync(path.join(process.cwd(), 'package-lock.json'), JSON.stringify(run.lock));
process.stdout.write(JSON.stringify(run.stdout || {added: 0}));
process.exit(run.exit || 0);
