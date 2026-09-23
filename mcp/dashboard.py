#!/usr/bin/env python
"""Review and edit the recipe corpus by hand, because a model cannot judge it.

WHY A HUMAN HAS TO BE IN THIS LOOP

The corpus is 938 recipes distilled from primary sources by language models.
They are good, and they are not trustworthy in the way a measured number is.
Three failure modes showed up in collection and none is detectable by the
thing that produced them:

  wrong           a subagent asserted that C23 REMOVES <stdnoreturn.h>. It
                  deprecates it. Caught only because a second pass re-read the
                  standard.
  overgeneralised "always use struct-of-arrays" is false half the time, and
                  the corpus contains the source that says so.
  unconditioned   26% of the first collections stated a trigger. Later ones
                  carry the `trigger_condition` field and the corpus now sits
                  at 46% (49% if the opening clause of the recipe counts).
                  The remainder claim to apply always, which for performance
                  advice is usually a missing condition rather than a
                  universal truth -- which is why the page can filter to
                  exactly those rows.

A domain expert reads one of these and knows in two seconds. No amount of
model review substitutes, because the reviewer shares the writer's blind
spots. So the corpus needs an editing surface, and the cheapest honest one is
a local page over the actual jsonl files.

WHAT IT DOES NOT DO

Score recipes. There is no "quality" field and no ranking, because nothing
here has measured which recipes help. `bench/recipe_oracle.py` and
`bench/mechanisms/hint_forms.py` exist to answer that, and until they have
run, a star rating would be an opinion wearing a number's clothes.

EDITS ARE WRITTEN BACK TO THE JSONL

Not to a database. The corpus is version-controlled text, a hand edit should
show up in `git diff` like any other change, and the review state lives in the
same row as the recipe so nothing has to be joined later.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPES = os.path.join(HERE, "..", "bench", "recipes")

sys.path.insert(0, HERE)

import dash_shell  # noqa: E402
import domains as domain_gate  # noqa: E402

# Review states. Deliberately few: anything finer would be a judgement the
# reviewer has to make repeatedly without a rule to make it by.
STATES = ("unreviewed", "keep", "edited", "reject")


def _files() -> list[str]:
    if not os.path.isdir(RECIPES):
        return []
    return sorted(f for f in os.listdir(RECIPES) if f.endswith(".jsonl"))


def load_all() -> list[dict]:
    """Every recipe, with a stable id of file plus line number.

    The id is positional because the source files have no key of their own.
    That is fine while rows are only ever edited in place, and `save_one`
    enforces that by writing back to the same line rather than appending.
    """
    out = []
    for fn in _files():
        path = os.path.join(RECIPES, fn)
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["_id"] = f"{fn}:{i}"
                r["_file"] = fn
                r["_domain"] = (r.get("language") or r.get("area")
                                or fn[:-6])
                # The domain tags the gate will actually apply. Declared and
                # inferred are kept apart rather than merged: a tag the
                # collector wrote is evidence, a tag mapped from `area` is a
                # guess this repo made, and a reviewer deciding whether to
                # tag a row by hand needs to know which one they are looking
                # at. `None` from the gate means untagged, which fails OPEN.
                doms = domain_gate.recipe_domains(r, fn)
                declared = [d for d in (r.get("domains") or [])
                            if d in domain_gate.DOMAINS]
                r["_domains"] = sorted(doms) if doms else []
                r["_domains_source"] = ("declared" if declared
                                        else "inferred" if doms else "none")
                r.setdefault("_state", "unreviewed")
                out.append(r)
    return out


def save_one(rid: str, changes: dict) -> dict:
    """Rewrite one row in place. Returns the stored row.

    The whole file is rewritten because jsonl lines vary in length and there
    is no safe in-place byte edit. At a few hundred rows that is free, and it
    keeps the on-disk format exactly what every other tool reads.
    """
    fn, _, idx = rid.rpartition(":")
    path = os.path.join(RECIPES, fn)
    if not os.path.exists(path):
        raise KeyError(f"no such file: {fn}")
    i = int(idx)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if i >= len(lines):
        raise KeyError(f"no such row: {rid}")
    row = json.loads(lines[i])

    # Only fields a reviewer is allowed to touch. The source URL and name are
    # provenance: if those are wrong the recipe should be rejected, not
    # quietly re-attributed to a source that does not say it.
    for k in ("recipe", "trigger_condition", "confidence", "_state", "_note"):
        if k in changes:
            row[k] = changes[k]
    if changes.get("recipe") and changes["recipe"] != json.loads(lines[i]).get("recipe"):
        row["_state"] = changes.get("_state") or "edited"
        row["_edited_at"] = int(time.time())
    lines[i] = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    row["_id"] = rid
    return row


def stats() -> dict:
    rows = load_all()
    by_state: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    by_domain_tag: dict[str, int] = {}
    untagged = 0
    over = 0
    conditioned = 0
    for r in rows:
        by_state[r["_state"]] = by_state.get(r["_state"], 0) + 1
        by_domain[r["_domain"]] = by_domain.get(r["_domain"], 0) + 1
        if r["_domains"]:
            for d in r["_domains"]:
                by_domain_tag[d] = by_domain_tag.get(d, 0) + 1
        else:
            untagged += 1
        if len((r.get("recipe") or "").split()) > 40:
            over += 1
        text = (r.get("recipe") or "").lower()
        if r.get("trigger_condition") or text.startswith(("if ", "when ")) \
                or " if " in text[:60] or " when " in text[:60]:
            conditioned += 1
    return {"total": len(rows), "by_state": by_state, "by_domain": by_domain,
            "by_domain_tag": by_domain_tag, "untagged_domains": untagged,
            "over_40_words": over, "with_trigger": conditioned,
            "without_trigger": len(rows) - conditioned,
            "files": _files()}


# The page. Chrome (tokens, head, nav) comes from `dash_shell` so this page
# and the vitals and results pages cannot drift apart; everything below is
# what is specific to reviewing recipes.
#
# Two things the restyle added, both because the corpus changed under it:
# domain tags, which decide whether a recipe is even a candidate for a task,
# and the trigger condition, which is the field the review exists to fill in.
STYLE = r"""<style>
main{max-width:1180px}
.lede{margin:var(--space-sm) 0 0;max-width:74ch;
  font-size:var(--text-small);color:var(--on-surface-variant)}
.stats{display:flex;flex-wrap:wrap;gap:var(--space-md) var(--space-xl)}
.stat{min-width:72px}
.stat .num{display:block}
.stat .label{display:block}
.filters{display:flex;flex-wrap:wrap;align-items:flex-end;
  gap:var(--space-md) var(--space-lg)}
.field{display:flex;flex-direction:column;gap:2px}
.field.check{flex-direction:row;align-items:center;gap:var(--space-sm)}
.field.check .label{color:var(--on-surface-variant);cursor:pointer}
.grow{flex:1 1 15rem;min-width:11rem}
.grow input{width:100%}

/* A row is a panel on the ground, so its edge may use --border (3.06:1).
   The state bar is the left edge widened, never a second line. */
.row{background:var(--surface-container-low);
  border:1px solid var(--border);
  border-left:3px solid var(--border-raised);
  border-radius:var(--radius);
  padding:var(--space-md) var(--space-lg);margin-bottom:var(--space-md)}
.row.is-keep{border-left-color:var(--primary-container)}
.row.is-edited{border-left-color:var(--tertiary-container)}
.row.is-reject{border-left-color:var(--secondary-container)}
/* A rejected row is de-emphasised by COLOUR at a measured 9.55:1, not by
   opacity. The old `.row.reject{opacity:.55}` dimmed the whole element, and
   the text survived it at 5.44:1 composited -- but the row's own border went
   with it, to #21252b at 1.23:1, so the rejected rows were the ones whose
   boundary you could no longer see. */
.row.is-reject textarea{color:var(--on-surface-variant)}

.meta{display:flex;flex-wrap:wrap;align-items:center;
  gap:var(--space-xs) var(--space-md);margin-bottom:var(--space-sm);
  font-size:var(--text-small);color:var(--on-surface-variant)}
.meta .src{margin-left:auto}
.rid{color:var(--outline)}
.words{font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);color:var(--outline)}
.words.is-over{color:var(--error)}
.badges{display:flex;flex-wrap:wrap;gap:var(--space-xs);
  margin-bottom:var(--space-sm)}
/* Inferred is dashed AND prefixed: never meaning by styling alone. */
.badge.is-inferred{border-style:dashed;color:var(--outline)}

/* The trigger is the point of the review, so it gets its own substrate and
   full-contrast body text rather than an accent it has no claim to. */
.trigger{background:var(--surface-container);
  border-left:2px solid var(--border-raised);
  border-radius:var(--radius);
  padding:var(--space-sm) var(--space-md);margin-bottom:var(--space-sm)}
.trigger .label{display:block}
.trigger p{margin:0}
.notrigger{margin-bottom:var(--space-sm)}

.actions{display:flex;flex-wrap:wrap;align-items:center;
  gap:var(--space-sm);margin-top:var(--space-sm)}
.saved{font-size:var(--text-small);color:var(--on-surface-variant)}
.saved.is-error{color:var(--error)}
.count{font-size:var(--text-label);font-weight:500;
  letter-spacing:var(--track-label);text-transform:uppercase;
  color:var(--outline)}
</style>
"""

BODY = r"""<main>
<div class="group">
  <h1 class="display">RECIPE CORPUS</h1>
  <p class="lede">Edits are written straight back to bench/recipes/*.jsonl, so
  a hand edit shows up in git diff like any other change. Provenance -- source
  name and URL -- is read-only: if a source does not say it, reject the row
  rather than re-attributing it. Nothing here is scored, because nothing has
  yet measured which recipes help.</p>
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

<section class="panel group" id="strip" aria-live="polite">
  <div class="panel-head">
    <span class="label">CORPUS SNAPSHOT</span>
    <span class="age" id="age"></span>
  </div>
  <div class="stats" id="stats"></div>
</section>

<section class="panel group" aria-label="filters">
  <div class="filters">
    <label class="field"><span class="label">DOMAIN</span>
      <select id="f-domain"><option value="">all</option></select></label>
    <label class="field"><span class="label">AREA</span>
      <select id="f-area"><option value="">all</option></select></label>
    <label class="field"><span class="label">STATE</span>
      <select id="f-state">
        <option value="">all</option><option>unreviewed</option>
        <option>keep</option><option>edited</option><option>reject</option>
      </select></label>
    <label class="field grow"><span class="label">SEARCH</span>
      <input id="f-q" placeholder="substring, any field"></label>
    <label class="field check">
      <input type="checkbox" id="f-untrig">
      <span class="label">ONLY WITHOUT A TRIGGER</span></label>
    <button id="reload" onclick="boot()">reload</button>
    <span class="count" id="count"></span>
  </div>
</section>

<div id="liststate"></div>
<div id="list"></div>
</main>

<script>
let ALL=[], STATS=null, LOADED=0, TICKER=null;
const $=id=>document.getElementById(id);
// The key lives in this browser only. It is the same key the editor sends;
// there is no separate dashboard account to create or remember.
const KEY=()=>localStorage.getItem('yamadori_key')||'';
const H=()=>({'Content-Type':'application/json',
              'Authorization':'Bearer '+KEY()});
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const words=s=>((s||'').trim()?(s||'').trim().split(/\s+/).length:0);
// A trigger is stated either in the field or in the opening clause of the
// recipe itself. Both count, because the corpus predates the field.
const hasTrig=r=>{const t=(r.recipe||'').toLowerCase().slice(0,60);
  return !!r.trigger_condition||t.startsWith('if ')||t.startsWith('when ')
    ||t.includes(' if ')||t.includes(' when ');};

/* ---- the five states ------------------------------------------------- */
function setState(kind,title,detail){
  const el=$('liststate');
  if(!kind&&!title){el.innerHTML='';return;}
  el.innerHTML='<div class="state '+(kind||'')+'">'+
    '<p class="state-title">'+esc(title)+'</p>'+
    '<p class="state-detail">'+esc(detail||'')+'</p></div>';
}
function showLogin(msg){
  $('login').hidden=false;
  // Everything on screen came from a read that is no longer authorised, so
  // none of it may stay: a stale count under a sign-in prompt is a number
  // presented as current that nothing is standing behind.
  if(TICKER){clearInterval(TICKER);TICKER=null;}
  LOADED=0; ALL=[];
  $('strip').classList.remove('is-stale');
  $('age').className='age'; $('age').textContent='';
  $('stats').innerHTML='';
  $('count').textContent='';
  $('list').innerHTML='';
  setState('','sign in to read the corpus',
    'The page is public because a browser cannot set an Authorization header '+
    'on a page load. Every /dash/api call is gated.');
  $('loginmsg').textContent=msg||'';
  $('loginmsg').className=msg?'saved is-error':'saved';
}
function saveKey(){
  localStorage.setItem('yamadori_key',$('key').value.trim());
  $('login').hidden=true;
  boot();
}

/* ---- staleness: a snapshot shown as current is an invented number ----- */
function tick(){
  if(!LOADED)return;
  const age=Math.round((Date.now()-LOADED)/1000);
  const stale=age>30;
  $('age').textContent=(stale?'stale - read ':'read ')+age+'s ago';
  $('age').classList.toggle('is-stale',stale);
  $('strip').classList.toggle('is-stale',stale);
}
function startTicking(){
  if(TICKER)clearInterval(TICKER);
  tick(); TICKER=setInterval(tick,1000);
}

/* ---- load ------------------------------------------------------------ */
async function get(path){
  const res=await fetch(path,{headers:H()});
  if(res.status===401){const e=new Error('unauthorised');e.auth=true;throw e;}
  if(!res.ok)throw new Error('HTTP '+res.status);
  return res.json();
}
async function boot(){
  $('login').hidden=true;
  setState('is-loading','reading bench/recipes/*.jsonl',
    'Every row is parsed from the jsonl on each load. There is no database '+
    'to be out of date with.');
  try{
    STATS=await get('/dash/api/stats');
  }catch(e){
    if(e.auth){showLogin('that key was not accepted');return;}
    setState('is-error','/dash/api/stats did not answer',
      e+' - this page is served by the yamadori server; check it is still '+
      'running on this port.');
    return;
  }
  renderStats(); fillFilters();
  try{
    ALL=await get('/dash/api/recipes');
  }catch(e){
    if(e.auth){showLogin('that key was not accepted');return;}
    setState('is-error','/dash/api/recipes did not answer',
      e+' - stats loaded, so the server is up and the recipe read itself '+
      'failed. Check bench/recipes/*.jsonl is readable.');
    return;
  }
  LOADED=Date.now(); startTicking(); render();
}

function stat(n,label){
  return '<div class="stat"><span class="num">'+n+'</span>'+
         '<span class="label">'+esc(label)+'</span></div>';
}
function renderStats(){
  const s=STATS, pct=n=>Math.round(n/Math.max(s.total,1)*100);
  $('stats').innerHTML=
    stat(s.total,'recipes')+
    stat(s.files.length,'files')+
    stat(s.with_trigger+' <span class="label">'+pct(s.with_trigger)+'%</span>',
         'state a trigger')+
    stat(s.without_trigger+' <span class="label">'+pct(s.without_trigger)+
         '%</span>','no trigger')+
    stat(s.over_40_words,'over 40 words')+
    stat(s.untagged_domains,'untagged')+
    Object.keys(s.by_state).sort().map(k=>stat(s.by_state[k],k)).join('');
}
function fillFilters(){
  const fill=(id,keys,extra)=>{
    const el=$(id), was=el.value;
    el.innerHTML='<option value="">all</option>'+
      keys.map(k=>'<option value="'+esc(k)+'">'+esc(k)+'</option>').join('')+
      (extra||'');
    el.value=was;
  };
  fill('f-domain',Object.keys(STATS.by_domain_tag).sort(),
       '<option value="~untagged">untagged</option>');
  fill('f-area',Object.keys(STATS.by_domain).sort());
}

/* ---- render ---------------------------------------------------------- */
const CHIP={keep:'chip chip-live',edited:'chip chip-flight',
            reject:'chip chip-cut',unreviewed:'chip'};
function badges(r){
  const kind=r._domains_source;
  if(!r._domains||!r._domains.length)
    return '<span class="badge is-inferred" title="no domain tags: this '+
      'recipe is eligible everywhere the gate is permissive">~ untagged</span>';
  const inf=kind==='inferred';
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
      (r.since_version?'<span>since '+esc(r.since_version)+'</span>':'')+
      (r.applies_to_version?'<span>'+esc(r.applies_to_version)+'</span>':'')+
      '<span class="words'+(w>40?' is-over':'')+'" id="w-'+cid+'">'+w+
        'w'+(w>40?' over 40':'')+'</span>'+
      '<span class="src">'+(r.source_url
        ? '<a href="'+esc(r.source_url)+'" target="_blank" rel="noopener">'+
          esc(r.source_name||r.source_url)+'</a>'
        : esc(r.source_name||''))+'</span>'+
    '</div>'+
    '<div class="badges">'+badges(r)+'</div>'+
    trig+
    '<textarea rows="3" aria-label="recipe text" data-id="'+id+'">'+
      esc(r.recipe)+'</textarea>'+
    '<div class="actions">'+
      '<button class="btn-primary" onclick="save(\'' + id + '\',\'keep\')">keep</button>'+
      '<button onclick="save(\'' + id + '\',\'edited\')">save edit</button>'+
      '<button class="btn-danger" onclick="save(\'' + id + '\',\'reject\')">reject</button>'+
      '<span class="saved" id="s-'+cid+'"></span>'+
    '</div></article>';
}
function render(){
  const dm=$('f-domain').value, ar=$('f-area').value, st=$('f-state').value;
  const q=$('f-q').value.toLowerCase(), ut=$('f-untrig').checked;
  const hits=ALL.filter(r=>
    (!ar||r._domain===ar)&&(!st||r._state===st)&&
    (!dm||(dm==='~untagged'?!(r._domains||[]).length
                           :(r._domains||[]).indexOf(dm)>=0))&&
    (!q||JSON.stringify(r).toLowerCase().includes(q))&&
    (!ut||!hasTrig(r)));
  const rows=hits.slice(0,300);
  $('count').textContent=hits.length>rows.length
    ? 'first '+rows.length+' of '+hits.length : hits.length+' shown';
  if(!ALL.length){
    $('list').innerHTML='';
    setState('','no recipes in bench/recipes/',
      'Nothing parsed out of the jsonl files. Check the directory exists and '+
      'holds one JSON object per line.');
    return;
  }
  if(!rows.length){
    $('list').innerHTML='';
    setState('','no rows match these filters',
      'Clear the search, widen the domain or area filter, or untick "only '+
      'without a trigger".');
    return;
  }
  setState(null);
  $('list').innerHTML=rows.map(rowHTML).join('');
}

/* ---- write straight back to the jsonl -------------------------------- */
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
    note.className='saved is-error';
    note.textContent='not written: '+j.error;
    return;
  }
  const row=ALL.find(r=>r._id===id);
  Object.assign(row,j.row);
  note.textContent='written to '+row._file;
  $('r-'+cid).className='row is-'+row._state;
  const w=words(row.recipe), we=$('w-'+cid);
  we.textContent=w+'w'+(w>40?' over 40':'');
  we.className='words'+(w>40?' is-over':'');
  const chip=$('r-'+cid).querySelector('.meta .chip');
  chip.className=CHIP[row._state]||'chip';
  chip.textContent=row._state;
}

['f-domain','f-area','f-state','f-q','f-untrig'].forEach(i=>{
  $(i).addEventListener('input',()=>{if(ALL.length)render();});
});
if(KEY())boot(); else showLogin();
</script>
"""

PAGE = dash_shell.head("Corpus") + STYLE + dash_shell.nav("corpus") + BODY

def handle_get(path: str):
    """(status, content_type, body_bytes) or None if this is not ours."""
    p = path.rstrip("/")
    if p == "/dash":
        return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")
    if p == "/dash/api/stats":
        return 200, "application/json", json.dumps(stats()).encode()
    if p == "/dash/api/recipes":
        return 200, "application/json", json.dumps(load_all()).encode()
    if p == "/dash/api/tiers":
        import tiers
        # `sent_effort` is what actually reaches the chat template: the
        # served template accepts only low/medium/xhigh, so `high` is sent as
        # `xhigh` and `minimal` as `low` (tiers.safe_effort, read from the
        # live template). Showing the requested value implied a setting the
        # model never receives.
        view = {n: dict(t, sent_effort=tiers.safe_effort(t["effort"]))
                for n, t in tiers.TIERS.items()}
        return 200, "application/json", json.dumps(
            {"order": tiers.ORDER, "tiers": view,
             "default": tiers.DEFAULT, "ceiling": tiers.CEILING}).encode()
    return None


def handle_post(path: str, body: dict):
    p = path.rstrip("/")
    if p != "/dash/api/recipe":
        return None
    try:
        row = save_one(body["id"], body)
        return 200, "application/json", json.dumps({"ok": True, "row": row}).encode()
    except Exception as e:                                       # noqa: BLE001
        return 400, "application/json", json.dumps(
            {"ok": False, "error": f"{type(e).__name__}: {e}"}).encode()


if __name__ == "__main__":
    s = stats()
    print(f"  {s['total']} recipes across {len(s['files'])} files")
    print(f"  {s['with_trigger']} state a trigger "
          f"({s['with_trigger'] * 100 // max(s['total'], 1)}%)")
    print(f"  {s['over_40_words']} over the 40-word limit")
    print(f"  review states: {s['by_state']}")
