#!/usr/bin/env python
"""The vitals page: what the hardware is holding, live, or plainly not live.

WHY THIS IS A PAGE AND NOT A PRINTOUT

`vitals.py` already prints everything here. It prints it once, at the moment
you remember to run it, which is the moment after the allocation failed. The
incident it was written for was visible for hours to anyone looking; nobody
was looking, because looking cost a terminal and a command.

WHAT THIS PAGE REFUSES TO DO

Show a number it did not measure. Every figure below comes out of one
`vitals.snapshot()` call and nothing is defaulted, padded or smoothed. A panel
with no rows renders the empty state and says which command would produce
some. The one place the payload is genuinely ambiguous -- `budget.pool_size()`
falls back to 131072 when the model server will not answer, and the snapshot
does not record which happened -- is called out in the page rather than papered
over.

Show a stale number as a current one. The snapshot carries `at`; the page
carries its age in seconds, in the header and next to the reload control, and
greys itself out past 30 seconds. A reading from four minutes ago presented as
now is the same lie as an invented one, told more convincingly.

HEADROOM IS THE HEADLINE

Per GPU the large figure is free MiB, not percent used. 85% used is fine. Two
gigabytes free when the next allocation is eight is already broken, and a
percentage is exactly the presentation that hides it.
"""
from __future__ import annotations

import json

import dash_shell
import vitals

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Vitals</title>
<style>
:root{
  /* Yamadori tokens. Borders use --outline-variant at FULL opacity: the
     token already carries the value that measures 3.06:1 against the ground,
     and compositing it at 35% lands at 1.38:1, which is the bug this
     replaced. */
  --bg:#0f131c; --lowest:#0a0e17; --low:#181c25; --raised:#1c2029;
  --high:#262a34; --highest:#31353f;
  --on-surface:#dfe2ef; --on-surface-variant:#b9cbb9; --outline:#849585;
  --outline-variant:#516854;
  --primary:#00ff88;      /* alive, healthy, primary action */
  --tertiary:#5cf2ff;     /* throughput and measurement in flight */
  --cut:#d4004b;          /* destructive, refused */
  --error:#ffb4ab;        /* a fault the operator must resolve */
  --xs:0.25rem; --sm:0.5rem; --md:0.75rem; --lg:1.25rem; --xl:2rem;
  --r:0.125rem; --rlg:0.25rem;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --display:Syne,'Segoe UI Variable Display',system-ui,sans-serif;
}
/* STALE: greying is done by re-pointing the tokens, never by opacity or a
   filter, so every resulting pair is still a known measured contrast. */
body.stale{
  --on-surface:#849585; --on-surface-variant:#849585;
  --primary:#849585; --tertiary:#849585; --error:#b9cbb9;
}
*{box-sizing:border-box}
body{margin:0;padding:var(--md);background:var(--bg);color:var(--on-surface);
  font:400 14px/1.6 var(--mono);letter-spacing:-0.01em;
  font-variant-numeric:tabular-nums}
h1{font:800 28px/1.1 var(--display);letter-spacing:-0.03em;margin:0}
h2{font:700 18px/1.2 var(--display);letter-spacing:0;margin:0}
a{color:var(--on-surface-variant)}
.label{font:500 11px/1.4 var(--mono);letter-spacing:0.06em;
  text-transform:uppercase;color:var(--outline)}
.small{font-size:12px;letter-spacing:0;color:var(--on-surface-variant)}
.num{font-variant-numeric:tabular-nums}

header{display:flex;flex-wrap:wrap;gap:var(--lg);align-items:baseline;
  justify-content:space-between;margin-bottom:var(--lg)}
header .sub{color:var(--on-surface-variant);font-size:12px;margin-top:var(--xs)}
.clock{display:flex;align-items:center;gap:var(--md);flex-wrap:wrap}
/* The single live element in the view. Glow lives here and nowhere else. */
.dot{width:10px;height:10px;border-radius:50%;background:var(--outline);
  flex:none}
.dot.live{background:var(--primary);box-shadow:0 0 8px 0 var(--primary)}
.dot.beat{animation:beat 600ms ease-out 1}
.dot.bad{background:var(--error);box-shadow:none}
@keyframes beat{from{transform:scale(1.9)}to{transform:scale(1)}}
button{font:inherit;color:var(--on-surface);background:var(--high);
  border:1px solid var(--outline-variant);border-radius:var(--r);
  padding:var(--sm) var(--md);min-height:2.75rem;cursor:pointer;
  transition:border-color 150ms ease-out,background 150ms ease-out}
button:hover{border-color:var(--primary)}
:focus-visible{outline:2px solid var(--tertiary);outline-offset:2px}
input{font:inherit;color:var(--on-surface);background:var(--raised);
  border:1px solid var(--outline-variant);border-radius:var(--r);
  padding:var(--sm) var(--md);min-height:2.75rem}

.panel{background:var(--low);border:1px solid var(--outline-variant);
  border-radius:var(--r);padding:var(--lg);margin-bottom:var(--lg)}
.phead{display:flex;gap:var(--md);align-items:baseline;justify-content:space-between;
  flex-wrap:wrap;margin-bottom:var(--md)}
.grid{display:grid;gap:var(--lg);grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.empty{color:var(--on-surface-variant);font-size:12px;line-height:1.7}
.empty code{background:var(--raised);border:1px solid var(--outline-variant);
  border-radius:var(--r);padding:0 var(--xs);color:var(--on-surface)}
.skel{color:var(--outline);font-size:12px;letter-spacing:0.06em}

/* GPUs: free memory is the figure, percent is a footnote. */
.gpu{border:1px solid var(--outline-variant);border-radius:var(--r);
  padding:var(--md);background:var(--raised);margin-bottom:var(--md)}
.gpu:last-child{margin-bottom:0}
.gpu .who{display:flex;gap:var(--sm);align-items:baseline;justify-content:space-between}
.free{display:flex;align-items:baseline;gap:var(--sm);margin:var(--sm) 0 var(--xs)}
.free .fig{font:600 28px/1 var(--mono);letter-spacing:-0.02em;color:var(--primary)}
.free .unit{color:var(--on-surface-variant);font-size:12px;letter-spacing:0.06em;
  text-transform:uppercase}
.gpu.tight .free .fig{color:var(--error)}
.meter{height:8px;background:var(--primary);border:1px solid var(--outline-variant);
  border-radius:var(--r);overflow:hidden}
.gpu.tight .meter{background:var(--error)}
.meter i{display:block;height:100%;background:var(--highest)}
.facts{display:flex;gap:var(--lg);flex-wrap:wrap;margin-top:var(--sm);
  color:var(--on-surface-variant);font-size:12px}

/* Context pool. */
.stats{display:flex;gap:var(--lg);flex-wrap:wrap}
.stat .v{font:600 20px/1.2 var(--mono);letter-spacing:-0.01em}
.split{display:flex;height:8px;margin-top:var(--md);border:1px solid var(--outline-variant);
  border-radius:var(--r);overflow:hidden}
.split .main{background:var(--primary)}
.split .helper{background:var(--tertiary)}
.split .reserve{background:var(--highest)}
.key{display:flex;gap:var(--lg);flex-wrap:wrap;margin-top:var(--sm);font-size:12px;
  color:var(--on-surface-variant)}
.key i{display:inline-block;width:10px;height:10px;border-radius:var(--r);
  margin-right:var(--xs);vertical-align:baseline}

table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font:500 11px/1.4 var(--mono);letter-spacing:0.06em;
  text-transform:uppercase;color:var(--outline);padding:0 var(--md) var(--sm) 0;
  border-bottom:1px solid var(--outline-variant)}
td{padding:var(--sm) var(--md) var(--sm) 0;border-bottom:1px solid var(--outline-variant);
  vertical-align:top}
tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right;padding-right:var(--lg)}

/* LOUD. A conflict is answered at random, which is the worst kind of wrong:
   it looks like it worked. Word plus colour plus rule, never colour alone. */
tr.conflict td{background:var(--high);border-bottom:1px solid var(--error);
  color:var(--error)}
tr.conflict td:first-child{border-left:3px solid var(--error)}
.badge{display:inline-block;font:500 11px/1.4 var(--mono);letter-spacing:0.06em;
  border:1px solid currentColor;border-radius:var(--r);padding:1px var(--xs);
  margin-left:var(--sm)}
.bad{color:var(--error)}
.ok{color:var(--primary)}
.why{color:var(--error);font-size:12px;letter-spacing:0}

.alarm{border:1px solid var(--error);border-left:3px solid var(--error);
  background:var(--high);border-radius:var(--r);padding:var(--md) var(--lg);
  margin-bottom:var(--lg)}
.alarm h2{color:var(--error)}
.alarm ul{margin:var(--sm) 0 0;padding-left:var(--lg)}
.alarm li{color:var(--error);margin-bottom:var(--xs)}
.quiet{border:1px solid var(--outline-variant);background:var(--low);
  border-radius:var(--r);padding:var(--md) var(--lg);margin-bottom:var(--lg);
  color:var(--on-surface-variant);font-size:12px}
.stalebar{border:1px solid var(--error);border-left:3px solid var(--error);
  background:var(--high);color:var(--error);border-radius:var(--r);
  padding:var(--md) var(--lg);margin-bottom:var(--lg);display:none}
body.stale .stalebar{display:block}
.note{color:var(--on-surface-variant);font-size:12px;margin-top:var(--sm);
  letter-spacing:0}

@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
}
</style>

<header>
  <div>
    <h1>VITALS</h1>
    <div class="sub">One page of measured hardware and process state. Every
      figure comes from a single <code>vitals.snapshot()</code>; nothing here is
      defaulted or smoothed. &middot; <a href="/dash">recipe corpus</a></div>
  </div>
  <div class="clock">
    <span class="dot" id="dot" aria-hidden="true"></span>
    <span id="age" class="small">starting</span>
    <button id="reload" type="button">measure now</button>
  </div>
</header>

<div class="panel" id="login" style="display:none">
  <h2>API key</h2>
  <div class="small" style="margin:var(--sm) 0 var(--md)">The same key your
    editor sends. It is kept in this browser only; there is no dashboard
    account.</div>
  <div style="display:flex;gap:var(--md);flex-wrap:wrap;align-items:center">
    <input id="key" type="password" size="36" placeholder="bearer key">
    <button type="button" id="signin">sign in</button>
    <span class="small" id="loginmsg"></span>
  </div>
</div>

<div id="errbar" class="alarm" style="display:none"></div>
<div class="stalebar" id="stalebar"></div>
<div id="warnbar"></div>

<main id="main">
  <div class="grid">
    <section class="panel" id="p-gpu">
      <div class="phead"><h2>GPUs</h2><span class="label">headroom</span></div>
      <div id="gpus" class="skel">measuring&hellip;</div>
    </section>
    <section class="panel" id="p-ctx">
      <div class="phead"><h2>Context pool</h2><span class="label">tokens</span></div>
      <div id="ctx" class="skel">measuring&hellip;</div>
    </section>
  </div>
  <div class="grid">
    <section class="panel" id="p-lis">
      <div class="phead"><h2>Listeners</h2><span class="label">ports are the truth</span></div>
      <div id="listeners" class="skel">measuring&hellip;</div>
    </section>
    <section class="panel" id="p-ep">
      <div class="phead"><h2>Endpoints</h2><span class="label">probed</span></div>
      <div id="endpoints" class="skel">measuring&hellip;</div>
    </section>
  </div>
  <section class="panel" id="p-seed">
    <div class="phead"><h2>Concept seed</h2><span class="label">last injected</span></div>
    <div id="seed" class="skel">measuring&hellip;</div>
  </section>
  <section class="panel" id="p-proc">
    <div class="phead"><h2>Processes</h2><span class="label" id="proccount"></span></div>
    <div id="dups"></div>
    <div id="procs" class="skel">measuring&hellip;</div>
  </section>
</main>

<script>
// The key lives in this browser only, and it is the same key the editor and
// the corpus page send. A browser cannot set an Authorization header on a
// page load, so the shell is public and every /dash/api call carries it.
const KEY=()=>localStorage.getItem('yamadori_key')||'';
const H=()=>({'Authorization':'Bearer '+KEY()});
const $=id=>document.getElementById(id);
const esc=s=>String(s===null||s===undefined?'':s)
  .replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const n=v=>(typeof v==='number'?v:Number(v)).toLocaleString('en-US');

const STALE_AFTER=30;   // seconds, from DESIGN.md
const EVERY=5000;       // poll interval; the next poll is scheduled after the
                        // previous one lands, because a snapshot shells out to
                        // netstat and PowerShell and can take several seconds.

let SNAP=null;          // the last snapshot that actually arrived
let RECEIVED=0;         // ms, local clock, when it arrived
let SKEW=null;          // server clock minus local clock at receipt, seconds
let ERR=null;           // current failure, named
let TIMER=null, INFLIGHT=false;

/* ---------------------------------------------------------------- states */

function ageSec(){
  if(!SNAP) return null;
  // Prefer the snapshot's own timestamp. If the server clock disagrees with
  // this browser's by more than two minutes, that difference is not age, so
  // fall back to measuring from receipt and say so.
  if(SKEW!==null && Math.abs(SKEW)>120)
    return Math.max(0,Math.round((Date.now()-RECEIVED)/1000));
  return Math.max(0,Math.floor(Date.now()/1000)-SNAP.at);
}

function tick(){
  const a=ageSec();
  const stale=a!==null&&a>STALE_AFTER;
  document.body.classList.toggle('stale',stale);
  const dot=$('dot');
  dot.className='dot'+(ERR?' bad':(SNAP&&!stale?' live':''));
  if(a===null){
    $('age').textContent=INFLIGHT?'measuring':'starting';
  }else{
    const skewed=(SKEW!==null&&Math.abs(SKEW)>120)?' (measured from receipt; '
      +'the server clock differs by '+n(SKEW)+'s)':'';
    $('age').innerHTML='measured <span class="num">'+a+'</span>s ago, at '+
      esc(new Date(SNAP.at*1000).toLocaleTimeString())+esc(skewed)+
      (INFLIGHT?' &middot; measuring&hellip;':'');
  }
  $('stalebar').innerHTML=a===null?'':
    '<strong>STALE</strong> &mdash; this page last measured '+a+' seconds ago'+
    ', past the '+STALE_AFTER+'s limit. Nothing below is current. '+
    (ERR?esc(ERR):'The poll is still running; if the age keeps climbing the '+
     'server at /dash/api/vitals has stopped answering.');
}

function fail(msg){
  ERR=msg;
  $('errbar').style.display='block';
  $('errbar').innerHTML='<h2>Vitals API error</h2><div class="small" '+
    'style="color:var(--error);margin-top:var(--sm)">'+esc(msg)+'</div>'+
    (SNAP?'<div class="note">The panels below still show the last snapshot '+
      'that arrived, and are marked with its age. They are not current.</div>'
     :'<div class="note">No snapshot has arrived yet, so nothing is shown '+
      'below rather than something plausible.</div>');
  tick();
}

function clearFail(){ERR=null;$('errbar').style.display='none';}

function showLogin(msg){
  $('login').style.display='block';
  $('loginmsg').textContent=msg||'';
  if(TIMER)clearTimeout(TIMER);
  TIMER=null;
}

/* ---------------------------------------------------------------- render */

function renderGPUs(g){
  const el=$('gpus');
  if(!g||!g.length){
    el.className='empty';
    el.innerHTML='No GPU was reported. <code>nvidia-smi</code> returned '+
      'nothing, so there is no memory figure to show. Check the driver is '+
      'loaded and that <code>nvidia-smi</code> is on PATH for the account '+
      'running the server.';
    return;
  }
  el.className='';
  el.innerHTML=g.map(x=>`
    <div class="gpu${x.tight?' tight':''}">
      <div class="who">
        <span><span class="label">GPU${esc(x.index)}</span>
          ${esc(x.name)}</span>
        ${x.tight?'<span class="badge bad">TIGHT</span>':''}
      </div>
      <div class="free">
        <span class="fig num">${n(x.free_mib)}</span>
        <span class="unit">MiB free</span>
      </div>
      <div class="meter" role="img" aria-label="${n(x.used_mib)} of ${n(x.total_mib)} MiB used, ${n(x.free_mib)} free">
        <i style="width:${x.pct}%"></i></div>
      <div class="facts">
        <span>used <span class="num">${n(x.used_mib)}</span> / <span class="num">${n(x.total_mib)}</span> MiB</span>
        <span><span class="num">${x.pct}</span>% used</span>
        <span>util <span class="num">${x.util}</span>%</span>
      </div>
      ${x.tight?'<div class="why">Free memory is below the threshold '+
        'vitals.py flags as tight: the next large allocation is the one that '+
        'fails.</div>':''}
    </div>`).join('');
}

function renderCtx(c){
  const el=$('ctx');
  if(!c||c.error){
    el.className='empty';
    el.innerHTML='No context budget. <code>budget.py</code> asks the model '+
      'server for <code>n_ctx</code> and it did not answer'+
      (c&&c.error?': <code>'+esc(c.error)+'</code>':'')+'. Start the stack '+
      'with <code>scripts/start-stack.bat</code>, then reload.';
    return;
  }
  if(!('pool' in c)){
    el.className='empty';
    el.innerHTML='The snapshot carried no pool size, so no budget can be '+
      'shown. Check <code>budget.pool_size()</code> against the running '+
      'server.';
    return;
  }
  const pct=v=>(100*v/Math.max(c.pool,1)).toFixed(2);
  el.className='';
  el.innerHTML=`
    <div class="stats">
      <div class="stat"><div class="label">pool</div><div class="v num">${n(c.pool)}</div></div>
      <div class="stat"><div class="label">main</div><div class="v num">${n(c.main)}</div></div>
      <div class="stat"><div class="label">helper, each</div><div class="v num">${n(c.helper)}</div></div>
      <div class="stat"><div class="label">reserve</div><div class="v num">${n(c.reserve)}</div></div>
      <div class="stat"><div class="label">KV</div><div class="v num">${c.gib} GiB</div></div>
    </div>
    <div class="split" role="img" aria-label="main ${n(c.main)}, helper ${n(c.helper)}, reserve ${n(c.reserve)} of ${n(c.pool)} tokens">
      <span class="main" style="width:${pct(c.main)}%"></span>
      <span class="helper" style="width:${pct(c.helper)}%"></span>
      <span class="reserve" style="width:${pct(c.reserve)}%"></span>
    </div>
    <div class="key">
      <span><i style="background:var(--primary)"></i>main, the conversation</span>
      <span><i style="background:var(--tertiary)"></i>deep thinking, one helper at a time</span>
      <span><i style="background:var(--highest)"></i>reserve, whatever the shares leave unclaimed</span>
    </div>
    ${c.pool===131072?'<div class="note">This pool is exactly 131072, which '+
      'is also the value <code>budget.pool_size()</code> falls back to when '+
      'the model server will not answer. The snapshot does not record which '+
      'of the two happened, so treat it as unconfirmed.</div>':''}`;
}

function renderListeners(rows){
  const el=$('listeners');
  if(!rows||!rows.length){
    el.className='empty';
    el.innerHTML='Nothing is listening on any port <code>vitals.py</code> '+
      'watches. Either the stack is down or it bound elsewhere. Start it with '+
      '<code>scripts/start-stack.bat</code>.';
    return;
  }
  el.className='';
  const conflicts=rows.filter(r=>r.conflict).length;
  el.innerHTML=`<table><thead><tr>
      <th class="n">port</th><th>role</th><th>address</th><th class="n">pid</th>
    </tr></thead><tbody>${rows.map(r=>`
      <tr class="${r.conflict?'conflict':''}">
        <td class="n num">${esc(r.port)}</td>
        <td>${esc(r.role)}${r.conflict?'<span class="badge">CONFLICT</span>':''}</td>
        <td>${esc(r.addr)}</td>
        <td class="n num">${esc(r.pid)}</td>
      </tr>`).join('')}</tbody></table>`+
    (conflicts?'<div class="why" style="margin-top:var(--md)">'+conflicts+
      ' listener row'+(conflicts===1?'':'s')+' share a port with another. '+
      'Requests to a contested port are answered at random by whichever '+
      'process the OS hands the connection &mdash; this is what served one '+
      'benchmark from two different builds. Kill the pid that should not be '+
      'there before trusting any measurement taken through it.</div>':'');
}

function renderEndpoints(rows){
  const el=$('endpoints');
  if(!rows||!rows.length){
    el.className='empty';
    el.innerHTML='No endpoint was probed. <code>vitals.endpoints()</code> '+
      'returned an empty list, which means the probe list itself is empty '+
      '&mdash; check <code>mcp/vitals.py</code>.';
    return;
  }
  el.className='';
  el.innerHTML=`<table><thead><tr>
      <th>name</th><th>state</th><th class="n">http</th><th class="n">ms</th>
    </tr></thead><tbody>${rows.map(e=>`
      <tr>
        <td>${esc(e.name)}</td>
        <td class="${e.ok?'ok':'bad'}">${e.ok?'answering':'DOWN'}</td>
        <td class="n num">${e.code?esc(e.code):'&mdash;'}</td>
        <td class="n num">${n(e.ms)}</td>
      </tr>`).join('')}</tbody></table>`+
    (rows.some(e=>!e.ok)?'<div class="why" style="margin-top:var(--md)">'+
      'A probe that did not answer reports http 0 and the time it spent '+
      'waiting, not a status.</div>':'');
}

function renderSeed(s){
  const el=$('seed');
  el.classList.remove('skel');
  if(!s){
    el.innerHTML='No seed has been put in a prompt yet. <code>concept_seed.phrase()</code> '+
      'records each one to <code>index/concept_seed_last.json</code> as it is injected.';
    return;
  }
  // The word, then the number that grows its limb: FNV-1a u32 in hex, the
  // same bits as binary, and the model token id it was drawn as.
  const bits=(s.u32>>>0).toString(2).padStart(32,'0').replace(/(.{8})(?!$)/g,'$1 ');
  const age=Math.max(0,Math.round(Date.now()/1000-s.at));
  el.innerHTML='<div class="num" style="font-size:1.6em;letter-spacing:.08em">'+esc(s.word)+'</div>'+
    '<div class="num">'+esc(s.hex)+'</div>'+
    '<div class="small num">'+bits+'</div>'+
    '<div class="small">token '+(s.token_id===null?'?':s.token_id)+' &middot; '+
    esc(s.where||'')+' &middot; <span class="num">'+age+'</span>s ago</div>';
}

function renderProcs(rows,dups){
  const el=$('procs');
  $('proccount').textContent=rows&&rows.length?rows.length+' matching':'';
  const d=$('dups');
  d.innerHTML=(dups&&dups.length)?'<div class="quiet">Duplicate process '+
    'trees: '+dups.map(x=>esc(x.what)+' (pids '+x.pids.map(esc).join(', ')+')')
    .join('; ')+'. Context, not an alarm &mdash; a second tree is only a '+
    'fault if it has also taken a port, which the listeners panel decides.'+
    '</div>':'';
  if(!rows||!rows.length){
    el.className='empty';
    el.innerHTML='No matching process. Nothing named python, llama-server or '+
      'llama-swap is running on this machine, so the stack is down. Start it '+
      'with <code>scripts/start-stack.bat</code>.';
    return;
  }
  el.className='';
  el.innerHTML=`<table><thead><tr>
      <th class="n">pid</th><th>what</th><th class="n">port</th>
    </tr></thead><tbody>${rows.map(p=>`
      <tr>
        <td class="n num">${esc(p.pid)}</td>
        <td>${esc(p.what)}</td>
        <td class="n num">${p.port?esc(p.port):'<span class="small">&mdash;</span>'}</td>
      </tr>`).join('')}</tbody></table>`;
}

function renderWarnings(ws){
  const el=$('warnbar');
  if(ws&&ws.length){
    el.innerHTML='<div class="alarm"><h2>'+ws.length+' warning'+
      (ws.length===1?'':'s')+'</h2><ul>'+
      ws.map(w=>'<li>'+esc(w)+'</li>').join('')+'</ul></div>';
  }else{
    el.innerHTML='<div class="quiet">No warnings: nothing measured in this '+
      'snapshot crossed a threshold. This is a statement about the snapshot '+
      'in the header, not about now.</div>';
  }
}

function render(){
  if(!SNAP) return;
  renderWarnings(SNAP.warnings);
  renderGPUs(SNAP.gpus);
  renderCtx(SNAP.context);
  renderListeners(SNAP.listeners);
  renderEndpoints(SNAP.endpoints);
  renderSeed(SNAP.seed);
  renderProcs(SNAP.processes,SNAP.duplicates);
  tick();
}

/* ------------------------------------------------------------------ poll */

function beat(){
  const d=$('dot');
  d.classList.remove('beat');
  void d.offsetWidth;
  d.classList.add('beat');
}

async function poll(){
  if(INFLIGHT) return;
  INFLIGHT=true;
  tick();
  try{
    const res=await fetch('/dash/api/vitals',{headers:H(),cache:'no-store'});
    if(res.status===401){
      INFLIGHT=false;
      showLogin('that key was not accepted by /dash/api/vitals');
      return;
    }
    if(!res.ok){
      fail('/dash/api/vitals answered HTTP '+res.status+'. The route is '+
           'served by mcp/dash_vitals.py through mcp/server.py; check the '+
           'server log for the traceback.');
    }else{
      const j=await res.json();
      if(j&&j.error){
        fail('vitals.snapshot() failed: '+j.error);
      }else{
        SNAP=j; RECEIVED=Date.now();
        SKEW=SNAP.at?(SNAP.at-Math.floor(RECEIVED/1000)):null;
        clearFail(); beat(); render();
      }
    }
  }catch(e){
    fail('Cannot reach /dash/api/vitals from this page ('+e+'). The dashboard '+
         'server is not answering; check that mcp/server.py is still running '+
         'on this port.');
  }finally{
    INFLIGHT=false;
    tick();
    if(TIMER)clearTimeout(TIMER);
    TIMER=setTimeout(poll,EVERY);
  }
}

$('signin').addEventListener('click',()=>{
  localStorage.setItem('yamadori_key',$('key').value.trim());
  $('login').style.display='none';
  poll();
});
$('key').addEventListener('keydown',e=>{if(e.key==='Enter')$('signin').click();});
$('reload').addEventListener('click',()=>{if(TIMER)clearTimeout(TIMER);poll();});
setInterval(tick,1000);
if(KEY()) poll(); else showLogin('');
</script>
"""


# The shared header, injected after this page's own stylesheet rather than
# baked into the template: the page was measured against the stylesheet above,
# and `nav_block` brings only the rules its own markup needs, scoped under
# `.shell-head`. Without this the three pages exist but nothing links them.
PAGE = PAGE.replace("</style>",
                    "</style>" + dash_shell.nav_block("vitals"), 1)



def handle_get(path: str):
    """(status, content_type, body_bytes) or None if this is not ours."""
    p = path.rstrip("/")
    if p == "/dash/vitals":
        return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")
    if p == "/dash/api/vitals":
        try:
            snap = vitals.snapshot()
        except Exception as e:                                   # noqa: BLE001
            # Named, not "something went wrong": the page prints this verbatim.
            return 500, "application/json", json.dumps(
                {"error": f"vitals.snapshot() raised {type(e).__name__}: {e}"}
            ).encode()
        return 200, "application/json", json.dumps(snap).encode()
    if p == "/dash/api/vitals/pulse":
        # The fast subset (vitals.pulse): slots, lanes, tools, queue, GPUs.
        # Cheap enough to poll every second; the tokonoma animates from it.
        try:
            snap = vitals.pulse()
        except Exception as e:                                   # noqa: BLE001
            return 500, "application/json", json.dumps(
                {"error": f"vitals.pulse() raised {type(e).__name__}: {e}"}
            ).encode()
        return 200, "application/json", json.dumps(snap).encode()
    return None


if __name__ == "__main__":
    s = vitals.snapshot()
    print(f"  /dash/vitals        {len(PAGE):>7} bytes of HTML")
    print(f"  /dash/api/vitals    {len(json.dumps(s)):>7} bytes of JSON, "
          f"{len(s['gpus'])} GPUs, {len(s['processes'])} processes, "
          f"{len(s['warnings'])} warnings")
