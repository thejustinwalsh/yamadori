#!/usr/bin/env python
"""The dataset manager: add a source, answer for it, watch it move, review it.

WHAT CHANGED AND WHY

`/dash` is where recipes are looked at. It is a good review surface and it is
the END of a pipeline whose other four stages had no surface at all. Adding a
corpus meant: find the source, run a collector by hand, remember to write the
licence into SOURCES.md, drop a jsonl into bench/recipes/, and hope somebody
reviews it. Every one of those steps was invisible while it was happening and
indistinguishable from not having started.

This page is that pipeline end to end:

    a PROMPT box        paste text, a URL, or say what you want added
    the QUESTIONS       whatever `datasets.missing()` says is still unanswered,
                        each carrying the reason it is being asked. Licence is
                        a blocker; "unknown" is not an answer.
    the PIPELINE        every dataset, its stage, and its jobs counted per
                        state. `errored` is drawn as its own thing, with the
                        error text and a re-run, because an errored job that
                        reads as done is the specific failure `jobs.py` was
                        written to prevent.
    the REVIEW          the existing keep / edit / reject surface, per dataset,
                        against the same POST /dash/api/recipe that /dash uses.

WHAT IT DOES NOT DO, AND SAYS SO

It does not run anything. `datasets.advance()` writes a job row; a worker
calling `jobs.claim()` is what would execute it, and THAT WORKER DOES NOT EXIST
YET. So the queue banner reports, from the database rather than from a hope,
whether any process has ever claimed a job here -- and when none has, it says
the queued jobs will not move. A spinner would have been a lie in a monospace
font.

NUMBERS

Job counts come from `COUNT(*) GROUP BY state`. Review counts come from
counting lines in the dataset's jsonl, and are ABSENT rather than zero when
that file does not exist. There is no progress bar anywhere on this page,
because nothing in the system can currently measure how far through a stage a
job is; `jobs.beat(progress=...)` carries a string, and that string is printed
verbatim when a worker sets one.

The review CSS is imported from `dashboard.STYLE` by reference rather than
copied, so the two surfaces cannot drift; `dashboard.py` itself is untouched.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dash_shell  # noqa: E402
import dashboard  # noqa: E402
import datasets  # noqa: E402

STYLE = r"""<style>
main{max-width:1180px}
.lede{margin:var(--space-sm) 0 0;max-width:78ch;
  font-size:var(--text-small);color:var(--on-surface-variant)}
.field{display:flex;flex-direction:column;gap:2px}
.grow{flex:1 1 15rem;min-width:11rem}
.grow input{width:100%}
.filters{display:flex;flex-wrap:wrap;align-items:flex-end;
  gap:var(--space-md) var(--space-lg)}
.actions{display:flex;flex-wrap:wrap;align-items:center;
  gap:var(--space-sm);margin-top:var(--space-sm)}
.saved{font-size:var(--text-small);color:var(--on-surface-variant)}
.saved.is-error{color:var(--error)}

/* ---- the prompt box ------------------------------------------------- */
.prompt textarea{min-height:5.5rem;font-size:var(--text-body)}
.prompt-row{display:flex;flex-wrap:wrap;align-items:flex-end;
  gap:var(--space-md);margin-top:var(--space-sm)}

/* ---- clarifying questions ------------------------------------------- */
/* A question that does not say why it is being asked gets answered by
   somebody who did not check, which looks identical to an answer. */
.q{background:var(--surface-container);
  border:1px solid var(--border-raised);
  border-left:3px solid var(--border-raised);
  border-radius:var(--radius);
  padding:var(--space-sm) var(--space-md);margin-bottom:var(--space-sm)}
.q.is-blocker{border-left-color:var(--error)}
.q-why{margin:2px 0 var(--space-sm);font-size:var(--text-small);
  color:var(--on-surface-variant);max-width:72ch}
.q input,.q select{width:100%}
.q .choices{display:flex;flex-wrap:wrap;gap:var(--space-xs) var(--space-md)}
.q .choices label{display:flex;align-items:center;gap:var(--space-xs);
  font-size:var(--text-small);color:var(--on-surface-variant)}

/* ---- the pipeline view ---------------------------------------------- */
.ds{background:var(--surface-container-low);
  border:1px solid var(--border);
  border-left:3px solid var(--border-raised);
  border-radius:var(--radius);
  padding:var(--space-md) var(--space-lg);margin-bottom:var(--space-md)}
/* The fault border is doubled by the word FAULT in the header: no meaning by
   colour alone, here or anywhere. */
.ds.is-fault{border-color:var(--error);border-left-color:var(--error)}
.ds.is-blocked{border-left-color:var(--secondary-container)}
.ds.is-running{border-left-color:var(--tertiary-container)}
.ds.is-complete{border-left-color:var(--primary-container)}
.ds-head{display:flex;flex-wrap:wrap;align-items:baseline;
  gap:var(--space-xs) var(--space-md);margin-bottom:var(--space-sm)}
.ds-head .title{margin-right:var(--space-xs)}
.ds-head .src{margin-left:auto;font-size:var(--text-small);
  color:var(--on-surface-variant)}
.did{color:var(--outline);font-size:var(--text-small)}

/* The stage track is a row of words, each either reached or not. It is not a
   progress bar: nothing here measures how far through a stage a job is. */
.track{display:flex;flex-wrap:wrap;gap:var(--space-xs);
  margin-bottom:var(--space-sm)}
.track .st{font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);text-transform:uppercase;
  padding:1px var(--space-sm);border-radius:var(--radius);
  border:1px solid var(--border-raised);color:var(--outline)}
.track .st.is-past{color:var(--on-surface-variant);
  border-color:var(--on-surface-variant)}
.track .st.is-here{color:var(--on-surface);border-color:var(--on-surface);
  background:var(--surface-container-high);font-weight:600}
.track .st.is-skipped{border-style:dashed}

.tallies{display:flex;flex-wrap:wrap;gap:var(--space-md) var(--space-lg);
  margin-bottom:var(--space-sm)}
.tally{min-width:60px}
.tally .num{display:block}
.tally .label{display:block}
.tally.is-fault .num{color:var(--error)}

.jobrow{display:flex;flex-wrap:wrap;align-items:baseline;
  gap:var(--space-xs) var(--space-md);
  padding:var(--space-xs) 0;font-size:var(--text-small);
  border-top:1px solid var(--border-raised);color:var(--on-surface-variant)}
.jobrow .jid{color:var(--outline)}
.err{margin:var(--space-xs) 0 0;padding:var(--space-sm) var(--space-md);
  background:var(--surface-container);
  border-left:2px solid var(--error);border-radius:var(--radius);
  font-size:var(--text-small);color:var(--on-surface);
  white-space:pre-wrap;word-break:break-word}
.note{margin:var(--space-xs) 0 0;font-size:var(--text-small);
  color:var(--on-surface-variant)}
.note.is-warn{color:var(--secondary)}
details.review{margin-top:var(--space-sm)}
details.review > summary{cursor:pointer;font-size:var(--text-label);
  font-weight:500;letter-spacing:var(--track-label);text-transform:uppercase;
  color:var(--on-surface-variant);min-height:24px}
details.review > summary:hover{color:var(--on-surface)}
.count{font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);text-transform:uppercase;
  color:var(--outline)}
</style>
"""

BODY = r"""<main>
<div class="group">
  <h1 class="display">DATASET MANAGER</h1>
  <p class="lede">Add a source, answer the questions it raises, and watch it
  move through submitted &rarr; clarify &rarr; extract &rarr;
  index. Every stage is a durable job row in index/jobs.sqlite3; this page
  enqueues them and never runs them. Licence is asked for and not guessed --
  this repo has already had to flag an AGPL corpus and a manual that prohibits
  redistribution. Review still writes straight back to
  bench/recipes/*.jsonl.</p>
</div>

<section class="panel group" id="login" hidden>
  <div class="panel-head"><span class="label">SIGN IN</span></div>
  <div class="filters">
    <label class="field grow"><span class="label">API KEY</span>
      <input id="key" type="password" autocomplete="off"
             placeholder="the same key your editor sends"></label>
    <button class="btn-primary" onclick="saveKey()">sign in</button>
    <span class="saved" id="loginmsg"></span>
  </div>
</section>

<section class="panel group prompt" aria-label="add a dataset">
  <div class="panel-head"><span class="label">ADD TO THE CORPUS</span></div>
  <label class="field"><span class="label">PROMPT, URL OR PASTED TEXT</span>
    <textarea id="p-prompt" rows="4"
      placeholder="a URL to read, text to distil, or what you want added"
      ></textarea></label>
  <div class="prompt-row">
    <label class="field grow"><span class="label">NAME (OPTIONAL)</span>
      <input id="p-name" placeholder="becomes bench/recipes/&lt;name&gt;.jsonl">
    </label>
    <label class="field"><span class="label">KIND</span>
      <select id="p-kind">
        <option value="recipes">recipes</option>
        <option value="laya">laya training set</option>
      </select></label>
    <button class="btn-primary" onclick="submitPrompt()">submit</button>
    <span class="saved" id="p-msg"></span>
  </div>
  <p class="note">A URL typed on its own is copied verbatim into the source
  URL. After the fetch, a model job (dataset.assist) proposes the source name,
  language and domains, and fills the licence ONLY from a verbatim quote it
  verified in the source or the site's LICENSE file; whatever it cannot
  establish is asked for next.</p>
</section>

<section class="panel group" id="queue" aria-live="polite">
  <div class="panel-head">
    <span class="label">QUEUE</span>
    <span class="age" id="age"></span>
  </div>
  <div class="tallies" id="qtallies"></div>
  <p class="note" id="qnote"></p>
</section>

<div id="pipestate"></div>
<div id="pipe"></div>
</main>

<script>
let DATA=null, RECIPES=null, LOADED=0, TICKER=null, OPEN=new Set();
const $=id=>document.getElementById(id);
const KEY=()=>localStorage.getItem('yamadori_key')||'';
const H=()=>({'Content-Type':'application/json',
              'Authorization':'Bearer '+KEY()});
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const words=s=>((s||'').trim()?(s||'').trim().split(/\s+/).length:0);
const hasTrig=r=>{const t=(r.recipe||'').toLowerCase().slice(0,60);
  return !!r.trigger_condition||t.startsWith('if ')||t.startsWith('when ')
    ||t.includes(' if ')||t.includes(' when ');};
const ago=t=>t?Math.round((Date.now()/1000)-t)+'s ago':'';

/* ---- the five states ------------------------------------------------- */
function setState(kind,title,detail){
  const el=$('pipestate');
  if(!kind&&!title){el.innerHTML='';return;}
  el.innerHTML='<div class="state '+(kind||'')+'">'+
    '<p class="state-title">'+esc(title)+'</p>'+
    '<p class="state-detail">'+esc(detail||'')+'</p></div>';
}
function showLogin(msg){
  $('login').hidden=false;
  if(TICKER){clearInterval(TICKER);TICKER=null;}
  LOADED=0; DATA=null; RECIPES=null;
  $('queue').classList.remove('is-stale');
  $('age').className='age'; $('age').textContent='';
  $('qtallies').innerHTML=''; $('qnote').textContent='';
  $('pipe').innerHTML='';
  setState('','sign in to read the pipeline',
    'The page is public because a browser cannot set an Authorization header '+
    'on a page load. Every /dash/api call is gated.');
  $('loginmsg').textContent=msg||'';
  $('loginmsg').className=msg?'saved is-error':'saved';
}
function saveKey(){
  localStorage.setItem('yamadori_key',$('key').value.trim());
  $('login').hidden=true; boot();
}

/* ---- staleness ------------------------------------------------------- */
function tick(){
  if(!LOADED)return;
  const age=Math.round((Date.now()-LOADED)/1000), stale=age>30;
  $('age').textContent=(stale?'stale - read ':'read ')+age+'s ago';
  $('age').classList.toggle('is-stale',stale);
  $('queue').classList.toggle('is-stale',stale);
}
function startTicking(){
  if(TICKER)clearInterval(TICKER);
  tick(); TICKER=setInterval(tick,1000);
}

/* ---- transport ------------------------------------------------------- */
async function get(path){
  const res=await fetch(path,{headers:H()});
  if(res.status===401){const e=new Error('unauthorised');e.auth=true;throw e;}
  if(!res.ok)throw new Error('HTTP '+res.status);
  return res.json();
}
async function post(path,body){
  const res=await fetch(path,{method:'POST',headers:H(),
                              body:JSON.stringify(body)});
  if(res.status===401){const e=new Error('unauthorised');e.auth=true;throw e;}
  return res.json();
}

async function boot(){
  $('login').hidden=true;
  setState('is-loading','reading the datasets table and the job queue',
    'Both live in index/jobs.sqlite3. Review counts are read from '+
    'bench/recipes/*.jsonl on each load.');
  try{
    DATA=await get('/dash/api/datasets');
  }catch(e){
    if(e.auth){showLogin('that key was not accepted');return;}
    setState('is-error','/dash/api/datasets did not answer',
      e+' - this page is served by the yamadori server; check it is still '+
      'running on this port, and that index/jobs.sqlite3 is writable.');
    return;
  }
  LOADED=Date.now(); startTicking(); renderQueue(); renderPipe();
}

/* ---- the queue banner ------------------------------------------------ */
function tallyHTML(n,label,fault){
  return '<div class="tally'+(fault&&n?' is-fault':'')+'">'+
    '<span class="num">'+n+'</span><span class="label">'+esc(label)+
    '</span></div>';
}
function renderQueue(){
  const q=DATA.queue.states, w=DATA.worker;
  $('qtallies').innerHTML=
    tallyHTML(q.queued||0,'queued')+
    tallyHTML(q.running||0,'running')+
    tallyHTML(q.done||0,'done')+
    tallyHTML(q.errored||0,'errored',true)+
    tallyHTML(q.cancelled||0,'cancelled')+
    tallyHTML(DATA.datasets.length,'datasets');
  const lanes=Object.keys(DATA.lanes).sort()
    .map(l=>l+' x'+DATA.lanes[l]).join('   ');
  if(!w.ever_claimed){
    $('qnote').className='note is-warn';
    $('qnote').textContent=
      'No process has ever claimed a job on this database. The worker loop '+
      'that calls jobs.claim() is not written yet, so queued jobs will not '+
      'move - submitting and advancing are recorded, nothing runs. '+
      'Lanes: '+lanes+'.';
  }else{
    $('qnote').className='note';
    $('qnote').textContent='Last claim '+ago(w.last_started)+
      ', last heartbeat '+ago(w.last_heartbeat)+'. Lanes: '+lanes+'.';
  }
}

/* ---- the pipeline view ----------------------------------------------- */
function trackHTML(d){
  const here=DATA.stages.indexOf(d.stage);
  return '<div class="track" aria-label="pipeline stages">'+
    DATA.stages.map((s,i)=>{
      const skipped=DATA.optional_stages.indexOf(s)>=0&&d.kind!=='laya';
      let cls='st';
      if(skipped)cls+=' is-skipped';
      else if(i<here)cls+=' is-past';
      else if(i===here)cls+=' is-here';
      return '<span class="'+cls+'" title="'+
        (skipped?'skipped: not a Laya training set'
               :i<here?'passed':i===here?'current stage':'not reached')+
        '">'+esc(s)+(skipped?' (skipped)':'')+'</span>';
    }).join('')+'</div>';
}
function questionsHTML(d){
  if(!d.missing.length)return '';
  const fid=esc(d.id);
  return '<form onsubmit="return answer(event,\''+fid+'\')">'+
    '<p class="label">'+d.missing.length+' UNANSWERED - THIS DATASET CANNOT '+
      'LEAVE CLARIFY</p>'+
    d.missing.map(m=>{
      const nm=esc(m.field);
      let ctl;
      if(m.input==='multi'){
        ctl='<div class="choices">'+(m.choices||[]).map(c=>
          '<label><input type="checkbox" name="'+nm+'" value="'+esc(c)+'"> '+
          esc(c)+'</label>').join('')+'</div>';
      }else{
        ctl='<input name="'+nm+'" data-ds="'+fid+'" '+
            'aria-label="'+esc(m.label)+'">';
      }
      return '<div class="q'+(m.severity==='blocker'?' is-blocker':'')+'">'+
        '<span class="label">'+esc(m.label)+
        (m.severity==='blocker'?' - BLOCKER':'')+'</span>'+
        '<p class="q-why">'+esc(m.why)+'</p>'+ctl+'</div>';
    }).join('')+
    '<div class="actions">'+
      '<button class="btn-primary" type="submit">answer</button>'+
      '<span class="saved" id="a-'+fid+'"></span>'+
    '</div></form>';
}
function runningHTML(d){
  // A running job says what its worker said, verbatim. jobs.beat(progress)
  // carries a string, not a fraction, so there is nothing here to draw a bar
  // from and none is drawn.
  return (d.running_jobs||[]).map(j=>
    '<div class="jobrow">'+
      '<span class="chip chip-flight">RUNNING</span>'+
      '<span class="jid">'+esc(j.id)+'</span>'+
      '<span>'+esc(j.queue)+'</span>'+
      '<span>lane '+esc(j.lane)+'</span>'+
      '<span>attempt '+j.attempts+' of '+j.max_attempts+'</span>'+
      '<span>started '+ago(j.started)+'</span>'+
      '<span>'+(j.progress?esc(j.progress)
                          :'the worker has set no progress string')+'</span>'+
    '</div>').join('');
}
function jobsHTML(d){
  if(!d.errored_jobs.length)return '';
  return d.errored_jobs.map(j=>
    '<div class="jobrow">'+
      '<span class="chip chip-fault">ERRORED</span>'+
      '<span class="jid">'+esc(j.id)+'</span>'+
      '<span>'+esc(j.queue)+'</span>'+
      '<span>lane '+esc(j.lane)+'</span>'+
      '<span>'+(j.attempts?j.attempts+' of '+j.max_attempts+' attempts'
                          :'errored before any attempt was claimed')+'</span>'+
      '<button class="btn-danger" onclick="rerun(\''+esc(j.id)+'\')">'+
        're-run this job</button>'+
      '<span class="saved" id="j-'+esc(j.id)+'"></span>'+
    '</div>'+
    '<p class="err">'+esc(j.error||'the job recorded no error text')+'</p>'
  ).join('');
}
function reviewSummary(d){
  if(d.review===null)
    return '<p class="note">No bench/recipes/'+esc(d.recipe_file)+' yet, so '+
      'there is nothing to review and no row count to report. Extract writes '+
      'that file.</p>';
  const keys=Object.keys(d.review).filter(k=>k!=='total'&&k!=='file');
  return '<div class="tallies">'+tallyHTML(d.review.total,'rows')+
    keys.sort().map(k=>tallyHTML(d.review[k],k)).join('')+'</div>'+
    '<details class="review" id="rev-'+esc(d.id)+'" '+
      (OPEN.has(d.id)?'open':'')+' ontoggle="toggled(\''+esc(d.id)+'\')">'+
    '<summary>review '+esc(d.recipe_file)+' - keep / edit / reject</summary>'+
    '<div id="rows-'+esc(d.id)+'"><div class="state is-loading">'+
      '<p class="state-title">not loaded</p>'+
      '<p class="state-detail">open this section to read the rows.</p>'+
      '</div></div></details>';
}
function dsHTML(d){
  const fault=d.jobs.errored>0;
  const cls='ds'+(fault?' is-fault'
    :d.missing.length?' is-blocked'
    :d.jobs.running?' is-running'
    :d.stage==='complete'?' is-complete':'');
  const warn=d.warnings.map(w=>
    '<p class="note is-warn">'+esc(w.kind.toUpperCase())+': '+esc(w.what)+
    '</p>').join('');
  // "enqueue" only where a row really goes into the queue; a human stage is
  // "move to", because calling it enqueue would describe work that is not
  // scheduled anywhere.
  const enq=DATA.enqueues[d.next_stage];
  const adv=d.next_stage&&!d.missing.length
    ? '<button class="btn-primary" onclick="advance(\''+esc(d.id)+'\')">'+
      (enq?'enqueue '+esc(d.next_stage)+' (lane '+esc(enq.lane)+')'
          :'move to '+esc(d.next_stage))+'</button>' : '';
  return '<article class="'+cls+'" id="ds-'+esc(d.id)+'">'+
    '<div class="ds-head">'+
      (fault?'<span class="chip chip-fault">FAULT</span>':'')+
      '<span class="title">'+esc(d.name)+'</span>'+
      '<span class="did">'+esc(d.id)+'</span>'+
      '<span class="chip">'+esc(d.kind)+'</span>'+
      '<span class="src">'+(d.source_url
        ? '<a href="'+esc(d.source_url)+'" target="_blank" rel="noopener">'+
          esc(d.source_name||d.source_url)+'</a>'
        : esc(d.source_name||'no source recorded'))+
        (d.licence?' - '+esc(d.licence):' - licence not established')+
      '</span>'+
    '</div>'+
    trackHTML(d)+
    '<div class="tallies">'+
      tallyHTML(d.jobs.total,'jobs')+
      tallyHTML(d.jobs.queued,'queued')+
      tallyHTML(d.jobs.running,'running')+
      tallyHTML(d.jobs.done,'done')+
      tallyHTML(d.jobs.errored,'errored',true)+
    '</div>'+
    runningHTML(d)+jobsHTML(d)+warn+questionsHTML(d)+reviewSummary(d)+
    '<div class="actions">'+adv+
      '<span class="saved" id="d-'+esc(d.id)+'"></span></div>'+
  '</article>';
}
function renderPipe(){
  if(!DATA.datasets.length){
    $('pipe').innerHTML='';
    setState('','no datasets yet',
      'Use the prompt box above. A submission is recorded in the datasets '+
      'table immediately; it does not need the GPU to exist.');
    return;
  }
  setState(null);
  $('pipe').innerHTML=DATA.datasets.map(dsHTML).join('');
  OPEN.forEach(id=>{if($('rev-'+id))loadRows(id);});
}
function toggled(id){
  const el=$('rev-'+id);
  if(!el)return;
  if(el.open){OPEN.add(id); loadRows(id);} else {OPEN.delete(id);}
}

/* ---- actions --------------------------------------------------------- */
async function submitPrompt(){
  const msg=$('p-msg');
  msg.className='saved'; msg.textContent='recording...';
  let j;
  try{
    j=await post('/dash/api/dataset',{prompt:$('p-prompt').value,
      name:$('p-name').value, kind:$('p-kind').value});
  }catch(e){
    if(e.auth){showLogin('that key was not accepted');return;}
    msg.className='saved is-error';
    msg.textContent='not recorded: '+e+' - the server did not answer';
    return;
  }
  if(!j.ok){
    msg.className='saved is-error'; msg.textContent='not recorded: '+j.error;
    return;
  }
  msg.textContent='recorded as '+j.dataset.id+' - answer its questions below';
  $('p-prompt').value=''; $('p-name').value='';
  boot();
}
async function answer(ev,id){
  ev.preventDefault();
  const note=$('a-'+id), body={id:id}, seen={};
  new FormData(ev.target).forEach((v,k)=>{
    if(seen[k]===undefined){seen[k]=v; body[k]=v;}
    else{body[k]=[].concat(body[k],v);}
  });
  // A checkbox group arrives one value per tick, so a single-tick answer
  // would otherwise be a bare string where the model expects a list.
  DATA.fields.forEach(f=>{
    if(f.input==='multi'&&body[f.name]!==undefined)
      body[f.name]=[].concat(body[f.name]);
  });
  note.className='saved'; note.textContent='saving...';
  let j;
  try{ j=await post('/dash/api/dataset/answer',body); }
  catch(e){
    if(e.auth){showLogin('that key was not accepted');return false;}
    note.className='saved is-error'; note.textContent='not saved: '+e;
    return false;
  }
  if(!j.ok){note.className='saved is-error';
            note.textContent='not saved: '+j.error; return false;}
  boot();
  return false;
}
async function advance(id){
  const note=$('d-'+id);
  note.className='saved'; note.textContent='enqueueing...';
  let j;
  try{ j=await post('/dash/api/dataset/advance',{id:id}); }
  catch(e){
    if(e.auth){showLogin('that key was not accepted');return;}
    note.className='saved is-error'; note.textContent='not enqueued: '+e;
    return;
  }
  if(!j.ok){
    note.className='saved is-error';
    note.textContent='refused: '+j.error+
      (j.reasons?' - '+j.reasons.map(r=>r.what).join('; '):'');
    return;
  }
  boot();
}
async function rerun(jid){
  const note=$('j-'+jid);
  note.className='saved'; note.textContent='requeueing...';
  let j;
  try{ j=await post('/dash/api/dataset/rerun',{job:jid}); }
  catch(e){
    if(e.auth){showLogin('that key was not accepted');return;}
    note.className='saved is-error'; note.textContent='not requeued: '+e;
    return;
  }
  if(!j.ok){note.className='saved is-error';
            note.textContent='refused: '+j.error; return;}
  boot();
}

/* ---- the existing review surface, per dataset ------------------------- */
/* Same endpoint, same fields, same rules as /dash: the row id is file:line,
   the POST is /dash/api/recipe, and provenance is read-only. */
const CHIP={keep:'chip chip-live',edited:'chip chip-flight',
            reject:'chip chip-cut',unreviewed:'chip'};
function badges(r){
  if(!r._domains||!r._domains.length)
    return '<span class="badge is-inferred" title="no domain tags: this '+
      'recipe is eligible everywhere the gate is permissive">~ untagged</span>';
  const inf=r._domains_source==='inferred';
  return r._domains.map(d=>'<span class="badge'+(inf?' is-inferred':'')+
    '" title="'+(inf?'inferred from this corpus&#39;s own area/language label'
                    :'declared on the recipe')+'">'+
    (inf?'~ ':'')+esc(d)+'</span>').join('');
}
function rowHTML(r){
  const w=words(r.recipe), id=esc(r._id), cid=CSS.escape(r._id);
  const trig=r.trigger_condition
    ? '<div class="trigger"><span class="label">TRIGGER CONDITION</span>'+
      '<p>'+esc(r.trigger_condition)+'</p></div>'
    : '<div class="notrigger"><span class="chip chip-fault">NO TRIGGER</span>'+
      '</div>';
  return '<article class="row is-'+esc(r._state)+'" id="r-'+cid+'">'+
    '<div class="meta">'+
      '<span class="'+(CHIP[r._state]||'chip')+'">'+esc(r._state)+'</span>'+
      '<span class="rid">'+id+'</span>'+
      '<span>'+esc(r._domain)+'</span>'+
      (r.category?'<span>'+esc(r.category)+'</span>':'')+
      (r.confidence?'<span>confidence '+esc(r.confidence)+'</span>':'')+
      '<span class="words'+(w>40?' is-over':'')+'" id="w-'+cid+'">'+w+
        'w'+(w>40?' over 40':'')+'</span>'+
      '<span class="src">'+(r.source_url
        ? '<a href="'+esc(r.source_url)+'" target="_blank" rel="noopener">'+
          esc(r.source_name||r.source_url)+'</a>'
        : esc(r.source_name||''))+'</span>'+
    '</div>'+
    '<div class="badges">'+badges(r)+'</div>'+trig+
    '<textarea rows="3" aria-label="recipe text" data-id="'+id+'">'+
      esc(r.recipe)+'</textarea>'+
    '<div class="actions">'+
      '<button class="btn-primary" onclick="save(\''+id+'\',\'keep\')">keep</button>'+
      '<button onclick="save(\''+id+'\',\'edited\')">save edit</button>'+
      '<button class="btn-danger" onclick="save(\''+id+'\',\'reject\')">reject</button>'+
      '<span class="saved" id="s-'+cid+'"></span>'+
    '</div></article>';
}
async function loadRows(id){
  const host=$('rows-'+id), d=DATA.datasets.find(x=>x.id===id);
  if(!host||!d)return;
  if(RECIPES===null){
    host.innerHTML='<div class="state is-loading">'+
      '<p class="state-title">reading bench/recipes/*.jsonl</p>'+
      '<p class="state-detail">Every row is parsed from the jsonl on each '+
      'load. There is no database to be out of date with.</p></div>';
    try{ RECIPES=await get('/dash/api/recipes'); }
    catch(e){
      if(e.auth){showLogin('that key was not accepted');return;}
      host.innerHTML='<div class="state is-error">'+
        '<p class="state-title">/dash/api/recipes did not answer</p>'+
        '<p class="state-detail">'+esc(e)+' - the datasets read succeeded, so '+
        'the server is up and the recipe read itself failed. Check '+
        'bench/recipes/*.jsonl is readable.</p></div>';
      return;
    }
  }
  const rows=RECIPES.filter(r=>r._file===d.recipe_file||r.dataset===d.id);
  if(!rows.length){
    host.innerHTML='<div class="state">'+
      '<p class="state-title">no rows for this dataset</p>'+
      '<p class="state-detail">Nothing in bench/recipes/ is tagged to '+
      esc(d.id)+' or lives in '+esc(d.recipe_file)+'. Extract writes that '+
      'file; it has not run.</p></div>';
    return;
  }
  host.innerHTML='<p class="count">'+rows.length+' rows</p>'+
    rows.slice(0,300).map(rowHTML).join('');
}
async function save(id,state){
  const cid=CSS.escape(id), note=$('s-'+cid);
  const ta=document.querySelector('textarea[data-id="'+cid+'"]');
  note.className='saved'; note.textContent='writing...';
  let j;
  try{
    const res=await fetch('/dash/api/recipe',{method:'POST',headers:H(),
      body:JSON.stringify({id:id,recipe:ta.value,_state:state})});
    if(res.status===401){showLogin('that key was not accepted');return;}
    j=await res.json();
  }catch(e){
    note.className='saved is-error';
    note.textContent='not written: '+e+' - the server did not answer';
    return;
  }
  if(!j.ok){
    note.className='saved is-error'; note.textContent='not written: '+j.error;
    return;
  }
  // dashboard.save_one returns the stored row with _id but WITHOUT _file --
  // it is derived from the id on load -- so the filename comes from the row
  // we already hold, exactly as /dash does it.
  const row=(RECIPES||[]).find(r=>r._id===id);
  if(row)Object.assign(row,j.row);
  note.textContent='written to '+((row&&row._file)||id.split(':')[0]);
  $('r-'+cid).className='row is-'+j.row._state;
  const w=words(j.row.recipe), we=$('w-'+cid);
  we.textContent=w+'w'+(w>40?' over 40':'');
  we.className='words'+(w>40?' is-over':'');
  const chip=$('r-'+cid).querySelector('.meta .chip');
  chip.className=CHIP[j.row._state]||'chip';
  chip.textContent=j.row._state;
}

if(KEY())boot(); else showLogin();
</script>
"""

# `dashboard.STYLE` is imported by reference, not copied: the review rows on
# this page are the same rows, and two copies of that CSS is how one of them
# quietly stops being the surface that was measured.
PAGE = (dash_shell.head("Data") + dashboard.STYLE + STYLE
        + dash_shell.nav("data") + BODY)


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload).encode()


def handle_get(path: str):
    """(status, content_type, body_bytes) or None if this path is not ours."""
    p = path.rstrip("/")
    if p == "/dash/data":
        return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")
    if p == "/dash/api/datasets":
        try:
            return _json(200, datasets.overview())
        except Exception as e:                                   # noqa: BLE001
            # Named, not "something went wrong": the page prints this verbatim.
            return _json(500, {"error": f"datasets.overview() raised "
                                        f"{type(e).__name__}: {e}"})
    if p.startswith("/dash/api/datasets/"):
        did = p.rsplit("/", 1)[-1]
        ds = datasets.get(did)
        if ds is None:
            return _json(404, {"error": f"no such dataset: {did}"})
        ds["recipe_file"] = datasets.recipe_file(ds)
        ds["jobs"] = datasets.job_counts(did)
        ds["job_rows"] = datasets.dataset_jobs(did)
        ds["review"] = datasets.review_counts(ds)
        ds["missing"] = datasets.missing(ds)
        ds["warnings"] = datasets.warnings(ds)
        ds["next_stage"] = datasets.next_stage(ds)
        # Each clarifying field with where its value came from: evidence
        # (verbatim quote, verified), proposed (the model's reading),
        # operator, or needs_you.
        ds["field_states"] = datasets.field_states(ds)
        ds["assist_jobs"] = [j for j in ds["job_rows"]
                             if j["queue"] == datasets.ASSIST[0]]
        return _json(200, ds)
    return None


def _submission(body: dict) -> tuple[str, str, dict]:
    """(prompt, source, fields) from either submission shape.

    The minimal one is {url} or {text} (or {prompt}), plus an optional kind:
    that is all the React form sends, and the assist does the rest. A URL is
    transcribed into source_url; text becomes the source itself. The classic
    page's full shape -- prompt, source and any answered fields -- still
    works unchanged.
    """
    fields = {k: body[k] for k in ("source_name", "source_url", "licence",
                                   "language", "domains", "notes")
              if k in body}
    url = str(body.get("url") or "").strip()
    text = str(body.get("text") or "")
    source = str(body.get("source") or "")
    if url:
        if not re.match(r"^https?://\S+$", url):
            raise ValueError(f"url {url[:120]!r} is not one http(s) URL; paste "
                             "the text instead")
        fields.setdefault("source_url", url)
    if text.strip() and not source.strip():
        source = text.strip()
    prompt = str(body.get("prompt") or "").strip() or url or \
        " ".join(text.split())[:400]
    return prompt, source, fields


def handle_post(path: str, body: dict):
    p = path.rstrip("/")
    if p not in ("/dash/api/dataset", "/dash/api/dataset/answer",
                 "/dash/api/dataset/advance", "/dash/api/dataset/rerun",
                 "/dash/api/dataset/assist"):
        return None
    body = body or {}
    try:
        if p == "/dash/api/dataset":
            prompt, source, fields = _submission(body)
            ds = datasets.create(
                prompt,
                name=body.get("name"),
                source=source,
                kind=body.get("kind") or "recipes",
                **fields)
            return _json(200, {"ok": True, "dataset": ds,
                               "missing": datasets.missing(ds),
                               "assist_jobs": [
                                   j for j in datasets.dataset_jobs(ds["id"])
                                   if j["queue"] == datasets.ASSIST[0]]})
        if p == "/dash/api/dataset/assist":
            # The operator asking the model again, explicitly. Never while
            # one is queued or running; never outside clarify.
            got = datasets.enqueue_assist(body.get("id"), force=True)
            if "skipped" in got:
                return _json(409, {"ok": False, "error": got["skipped"],
                                   "job": got.get("job")})
            return _json(200, {"ok": True, "job": got["job"]})
        if p == "/dash/api/dataset/answer":
            did = body.get("id")
            ds = datasets.answer(did, {k: v for k, v in body.items()
                                       if k != "id"})
            return _json(200, {"ok": True, "dataset": ds,
                               "missing": datasets.missing(ds),
                               "warnings": datasets.warnings(ds)})
        if p == "/dash/api/dataset/advance":
            ds = datasets.advance(body.get("id"), to=body.get("to"))
            return _json(200, {"ok": True, "dataset": ds,
                               "enqueued": ds.get("enqueued")})
        ds = datasets.rerun(body.get("job"))
        return _json(200, {"ok": True, "job": ds})
    except datasets.Blocked as e:
        # A refusal with its reasons attached. The page prints each `what`,
        # because "cannot advance" without the why is a dead end.
        return _json(409, {"ok": False, "error": str(e),
                           "stage": e.stage, "reasons": e.reasons})
    except Exception as e:                                       # noqa: BLE001
        return _json(400, {"ok": False,
                           "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    o = datasets.overview()
    print(f"  /dash/data          {len(PAGE):>7} bytes of HTML")
    print(f"  /dash/api/datasets  {len(json.dumps(o)):>7} bytes of JSON, "
          f"{len(o['datasets'])} dataset(s)")
    if not o["worker"]["ever_claimed"]:
        print("  no worker has ever claimed a job here; queued jobs will "
              "not move")
