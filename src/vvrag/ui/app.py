"""Localhost UI for altVRAG pipelines (stdlib only).

    python -m vvrag.ui.app [--port 8080] [--index-dir data/ui_index]

Binds 127.0.0.1 only. Keys are written to .env (gitignored, never logged).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

INDEX_DIR = "data/ui_index"
WORK_DIR = "results/ui_work"
MAX_BODY = 300 * 1024 * 1024

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>altVRAG — local console</title>
<style>
body{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222}
.card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:14px 0}
h2{margin:0 0 10px} label{display:block;margin:8px 0 2px;font-weight:600}
input[type=text],input[type=password],input[type=number],select,textarea{width:100%;padding:8px;border:1px solid #ccc;border-radius:6px}
button{background:#1a73e8;color:#fff;border:0;border-radius:6px;padding:9px 16px;cursor:pointer;margin-top:10px}
button:disabled{background:#999} .row{display:flex;gap:12px} .row>div{flex:1}
#status{color:#555;font-size:14px} .badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:13px}
.ok{background:#e6f4ea}.warn{background:#fef7e0}.bad{background:#fce8e6}
pre{background:#f6f6f6;padding:10px;border-radius:6px;overflow:auto;font-size:13px}
img.ev{max-width:100%;border:1px solid #ddd;border-radius:6px;margin-top:8px}
.small{font-size:13px;color:#555}
</style></head><body>
<h1>altVRAG — local console</h1>
<div id="status" class="card">loading status…</div>

<div class="card"><h2>1. API keys &amp; reader</h2>
<div class="row"><div><label>Gemini key</label><input type="password" id="k_gemini" placeholder="GEMINI_API_KEY"></div>
<div><label>Groq key</label><input type="password" id="k_groq" placeholder="GROQ_API_KEY"></div>
<div><label>OpenRouter key</label><input type="password" id="k_openrouter" placeholder="OPENROUTER_API_KEY"></div></div>
<div class="row"><div><label>Provider</label><select id="provider"><option>gemini</option><option>groq</option><option>openrouter</option></select></div>
<div><label>Model (free multimodal, see provider docs)</label><input type="text" id="model" placeholder="e.g. gemini-2.0-flash"></div></div>
<button onclick="saveKeys()">Save keys &amp; model</button> <span id="keys_msg" class="small"></span>
<p class="small">Stored in <code>.env</code> (gitignored, localhost only). Empty fields leave existing values untouched.</p></div>

<div class="card"><h2>2. Upload PDFs / page images</h2>
<input type="file" id="files" multiple accept=".pdf,.png,.jpg,.jpeg,.tif,.bmp">
<label><input type="checkbox" id="skip_ocr" checked> skip OCR (no Tesseract on most laptops; needs text-layer PDFs)</label><br>
<button onclick="upload()">Upload &amp; ingest</button> <span id="up_msg" class="small"></span>
<pre id="manifest"></pre></div>

<div class="card"><h2>3. Ask a question</h2>
<label>Question</label><textarea id="q" rows="2" placeholder="What is reported for Aurora?"></textarea>
<div class="row">
<div><label>Reader</label><select id="reader_mode"><option value="mock">mock (canned, no key)</option><option value="live">live VLM (needs key)</option></select></div>
<div><label>Samples</label><input type="number" id="n_samples" value="3" min="1" max="5"></div>
<div><label>Grounding</label><select id="grounding"><option value="span">keyword span (honest)</option><option value="heatmap">heatmap demo (seeded C1)</option></select></div>
<div><label>Gate</label><select id="gate"><option value="none">none</option><option value="fixed">fixed tau</option><option value="calibrated">calibrated (scorer+tau)</option><option value="conformal">conformal (tau file)</option></select></div>
<div><label>Fixed tau</label><input type="number" id="fixed_tau" value="0.5" step="0.05" min="0" max="1"></div>
</div>
<button onclick="ask()">Ask</button> <span id="ask_msg" class="small"></span>
<div id="result"></div></div>

<p class="small">Pipeline: retrieval is <b>keyword overlap</b> (never abstains; the MaxSim query
encoder runs on GPU only) &rarr; the reader is shown <b>numbered sentences</b> and cites
<code>sentence_ids</code> + a verbatim <code>quote</code> &rarr; a <b>deterministic quote check</b>
(no model) + statement NLI verify the citation &rarr; the confidence gate decides; boxes are drawn on
the <b>cited sentences only</b> (else "not found in this document", with closest passages).
Gate "calibrated/conformal" uses <code>data/gate/scorer.pkl</code> + <code>tau.json</code> from
<code>python scripts/calibrate_gate.py</code>. Full numbers: <code>python -m vvrag.eval.run_eval --all</code>.</p>

<script>
async function jget(u){const r=await fetch(u);return r.json()}
async function jpost(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});const t=await r.json();if(!r.ok)throw new Error(t.error||r.status);return t}
async function refresh(){
 const s=await jget('/api/status');
 if(!window._ocrInit && s.ocr_available){
  document.getElementById('skip_ocr').checked=false; window._ocrInit=true;
  document.getElementById('up_msg').textContent='Tesseract found — OCR enabled';
 }
 document.getElementById('status').innerHTML=
  `<b>Index:</b> ${s.docs} docs / ${s.pages} pages &nbsp;|&nbsp; <b>SDKs:</b> gemini:${s.sdk.gemini} groq:${s.sdk.groq} openrouter:${s.sdk.openrouter}`+
  ` &nbsp;|&nbsp; <b>Keys set:</b> gemini:${s.keys.gemini} groq:${s.keys.groq} openrouter:${s.keys.openrouter}`+
  ` &nbsp;|&nbsp; <b>Reader:</b> ${s.reader.provider}/${s.reader.model}`;
}
async function saveKeys(){
 const b={gemini_key:document.getElementById('k_gemini').value,groq_key:document.getElementById('k_groq').value,
  openrouter_key:document.getElementById('k_openrouter').value,provider:document.getElementById('provider').value,
  model:document.getElementById('model').value};
 try{await jpost('/api/keys',b);document.getElementById('keys_msg').textContent='saved';refresh()}
 catch(e){document.getElementById('keys_msg').textContent='error: '+e.message}
}
async function upload(){
 const fs=document.getElementById('files').files;
 if(!fs.length){document.getElementById('up_msg').textContent='pick files first';return}
 const msg=document.getElementById('up_msg');
 try{
  const arr=[];
  for(const f of fs){
   msg.textContent=`reading ${f.name} (${Math.round(f.size/1024)} KB)…`;
   const b=await f.arrayBuffer();
   arr.push({name:f.name,b64:u8ToB64(new Uint8Array(b))});
  }
  msg.textContent='uploading…';
  const m=await jpost('/api/upload',{files:arr,skip_ocr:document.getElementById('skip_ocr').checked});
  msg.textContent='ingesting on server…';
  document.getElementById('manifest').textContent=JSON.stringify(m,null,1);
  msg.textContent='done';refresh();
 }catch(e){msg.textContent='error: '+(e&&e.message?e.message:e)}
}
function u8ToB64(u8){
 // chunked: spreading the whole array into apply() blows the arg limit (RangeError)
 let s='';const CH=0x8000;
 for(let i=0;i<u8.length;i+=CH){s+=String.fromCharCode.apply(null,u8.subarray(i,Math.min(i+CH,u8.length)))}
 return btoa(s);
}
async function ask(){
 const q=document.getElementById('q').value.trim();
 if(!q){document.getElementById('ask_msg').textContent='type a question';return}
 document.getElementById('ask_msg').textContent='thinking… (first call loads NLI, ~30s)';
 const b={question:q,reader_mode:document.getElementById('reader_mode').value,
  n_samples:+document.getElementById('n_samples').value,grounding:document.getElementById('grounding').value,
  gate:document.getElementById('gate').value,fixed_tau:+document.getElementById('fixed_tau').value,
  provider:document.getElementById('provider').value};
 try{const r=await jpost('/api/ask',b);
  document.getElementById('ask_msg').textContent=`done in ${r.timings.total}s`;
  render(r);
 }catch(e){document.getElementById('ask_msg').textContent='error: '+e.message}
}
function render(r){
 const ev=r.evidence||{};
 const cls=r.abstained?'bad':'ok';
 const scoreBadge=r.abstained
  ?`<span class="badge" title="confidence score vs the abstain threshold">score ${r.score}${r.gate_mode&&r.gate_mode!=='none'?(' / tau '+r.tau):''}</span>`
  :`<span class="badge ok">score ${r.score}${r.gate_mode&&r.gate_mode!=='none'?(' / tau '+r.tau):''}</span>`;
 let html=`<h3>Answer <span class="badge ${cls}">${r.abstained?'NOT FOUND':'ANSWERED'}</span> ${scoreBadge}</h3>`;
 if(r.abstained){
  html+=`<p class="badge warn">Not found in this document.</p>`;
  if(r.abstain_reason)html+=`<p class="small">reason: ${esc(r.abstain_reason)}</p>`;
  if((r.closest_passages||[]).length){
   html+=`<p class="small"><b>Closest passages:</b></p><ul>`;
   for(const p of r.closest_passages){html+=`<li>${esc(p.text).slice(0,240)} <span class="small">(score ${p.score})</span></li>`}
   html+=`</ul>`;
  }
 } else {
  const tierOk=r.tier==='text'||r.verified===true;
  const vBadge=tierOk?'':(r.tier==='visual'
   ?'<span class="badge warn" title="answer shown by a visual region; confirmed by an independent cross-family check (not a verbatim text quote)">not text-verified (visual region)</span>'
   :'<span class="badge warn" title="answer not contained in / entailed by the cited text">evidence unverified</span>');
  html+=`<p><b>${esc(r.answer)}</b> ${vBadge}</p>`;
  const quote=(r.samples&&r.samples[0]&&r.samples[0].quote)||'';
  if(quote)html+=`<p class="small"><b>quote:</b> \u201c${esc(quote)}\u201d</p>`;
  if(r.support&&r.support.require)html+=`<p class="small">answer-support: ${esc(r.support.method)} (score ${r.support.score})</p>`;
  if(r.citations&&r.citations.status&&r.citations.status!=='none'&&r.citations.status!=='ok')
   html+=`<p class="small">citations: ${esc(r.citations.status)} (${r.citations.n_resolved}/${r.citations.n_refs} resolved)</p>`;
 }
 html+=`<p class="small">page: ${ev.doc_id||'-'} p${ev.page??'-'} | crop text: ${esc((ev.crop_text||'')).slice(0,300)}</p>`;
 const cited=r.cited_sentences||[];
 // Phase 2: citations may live on several pages -> page tabs. Legacy results
 // without ev.pages behave exactly like before (single primary page).
 let pgs=(ev.pages&&ev.pages.length)?ev.pages.slice():
  [{doc_id:ev.doc_id,page:ev.page,page_url:ev.page_url,box_frac:ev.box_frac||[]}];
 pgs.forEach(function(p){
  p.cites=cited.map(function(c,i){return {i:i,c:c}}).filter(function(x){
   const pgOk=(x.c.page==null)?(p.page===ev.page&&p.doc_id===ev.doc_id):(x.c.page===p.page);
   return pgOk&&(x.c.doc_id==null||x.c.doc_id===p.doc_id);
  }).map(function(x){return {i:x.i,frac:x.c.box_frac,sid:x.c.sent_id,text:x.c.text}});
 });
 const score=function(p){return p.cites.length+((p.doc_id===ev.doc_id&&p.page===ev.page)?0.5:0)};
 let active=0,bestv=-1;
 pgs.forEach(function(p,k){const v=score(p);if(v>bestv){bestv=v;active=k}});
 if(pgs.length>1){
  html+=`<p class="small"><b>cited pages:</b> `;
  pgs.forEach(function(p,k){
   html+=`<button class="ptab" data-k="${k}" style="margin-right:6px;cursor:pointer">`
    +`${esc(p.doc_id||'')} p${p.page}</button>`;
  });
  html+=`</p>`;
 }
 if(ev.overlay_url||pgs.length){
  html+=`<div id="evwrap" style="position:relative;display:inline-block;max-width:100%">`
   +`<img class="ev" id="evimg"></div>`;
  html+=`<div id="cropwrap" style="display:${pgs.length>1?'none':''}">`
   +`<img class="ev" id="evcrop" src="${ev.crop_url||''}"></div>`;
 }
 if(cited.length){
  html+=`<p class="small"><b>Cited sentences</b> (click to highlight):</p><ul id="sentlist">`;
  cited.forEach(function(s,i){
   const tag=s.sent_id<0?`<span class="badge warn">region</span>`
    :`<span class="badge ok">S${s.sent_id}${s.page!=null?(' p'+s.page):''}</span>`;
   html+=`<li class="sitem" data-i="${i}" style="cursor:pointer">`
    +`${tag} ${esc(s.text)}</li>`;
  });
  html+=`</ul>`;
 }
 html+=`<pre>${esc(JSON.stringify({features:r.features,retrieval:r.retrieval,api_calls:r.api_calls},null,1))}</pre>`;
 document.getElementById('result').innerHTML=html;
 LAST_EV=ev; PAGES=pgs;
 setPage(active);
 document.querySelectorAll('.ptab').forEach(function(b){b.onclick=function(){setPage(+b.dataset.k)}});
 document.querySelectorAll('#sentlist .sitem').forEach(function(li){
  li.onclick=function(){highlight(+li.getAttribute('data-i'))};
 });
}
let PAGES=[],activePage=0,LAST_EV=null;
function setPage(k){
 const pg=PAGES[k];if(!pg)return;activePage=k;
 const ev=LAST_EV||{};
 const img=document.getElementById('evimg');if(!img)return;
 const isPrim=(pg.doc_id===ev.doc_id&&pg.page===ev.page);
 img.src=(isPrim&&ev.overlay_url)?ev.overlay_url:(pg.page_url||ev.page_url||'');
 document.querySelectorAll('.ptab').forEach(function(b){
  const on=(+b.dataset.k===k);b.style.background=on?'#1a73e8':'';b.style.color=on?'#fff':'';
 });
 const cw=document.getElementById('cropwrap');
 if(cw)cw.style.display=(isPrim&&ev.crop_url)?'':'none';
 const wrap=document.getElementById('evwrap');if(!wrap)return;
 wrap.querySelectorAll('.sentbox').forEach(function(b){b.remove()});
 (pg.cites||[]).forEach(function(ci){
  if(!ci.frac)return;
  const d=document.createElement('div');d.className='sentbox';d.dataset.i=ci.i;
  d.title='S'+ci.sid+': '+(ci.text||'');
  d.style.position='absolute';d.style.border='2px solid #1a73e8';d.style.boxSizing='border-box';
  d.style.pointerEvents='none';d.style.display='none';
  d.style.left=(ci.frac[0]*100)+'%';d.style.top=(ci.frac[1]*100)+'%';
  d.style.width=((ci.frac[2]-ci.frac[0])*100)+'%';d.style.height=((ci.frac[3]-ci.frac[1])*100)+'%';
  wrap.appendChild(d);
 });
 img.onload=function(){wrap.style.width=img.clientWidth+'px'};
 if(img.complete&&img.naturalWidth)img.onload();
}
function highlight(i){
 const k=PAGES.findIndex(function(p){return (p.cites||[]).some(function(ci){return ci.i===i})});
 if(k>=0&&k!==activePage)setPage(k);
 document.querySelectorAll('#evwrap .sentbox').forEach(function(b){b.style.display=(+b.dataset.i===i)?'block':'none'});
 document.querySelectorAll('#sentlist .sitem').forEach(function(li){
  li.style.background=(+li.getAttribute('data-i')===i)?'#fff3c4':'';
 });
}
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c]})}
refresh();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "altVRAG/0.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > MAX_BODY:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/status":
            self._json(status())
        elif parsed.path.startswith("/media/"):
            name = Path(parsed.path[len("/media/"):]).name
            for base in (self.server.work_dir, str(Path(self.server.index_dir) / "pages")):
                p = Path(base) / name
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    data = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self._json({"error": "not found"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/api/keys":
                self._json(save_keys(self._body()))
            elif self.path == "/api/upload":
                self._json(do_upload(self._body(), self.server.index_dir))
            elif self.path == "/api/ask":
                b = self._body()
                from vvrag.config import load_config

                cfg = load_config("config.yaml")
                self._json(answer_ui(b, self.server.index_dir, self.server.work_dir,
                                     cfg.reader.provider))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def sdk_present(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def status() -> dict:
    from vvrag.config import load_config

    cfg = load_config("config.yaml")
    docs = pages = 0
    try:
        import sqlite3

        con = sqlite3.connect(Path(INDEX_DIR) / "meta.db")
        docs = con.execute("SELECT COUNT(DISTINCT doc_id) FROM pages").fetchone()[0]
        pages = con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    except Exception:
        pass
    out = {
        "docs": docs, "pages": pages,
        "sdk": {"gemini": sdk_present("google.genai"),
                "groq": sdk_present("groq"), "openrouter": sdk_present("openai")},
        "keys": {"gemini": bool(os.environ.get("GEMINI_API_KEY")),
                 "groq": bool(os.environ.get("GROQ_API_KEY")),
                 "openrouter": bool(os.environ.get("OPENROUTER_API_KEY"))},
        "reader": {"provider": cfg.reader.provider, "model": cfg.reader.model},
    }
    try:
        from vvrag.ingest.ocr import tesseract_available

        out["ocr_available"] = tesseract_available()
    except Exception:
        out["ocr_available"] = False
    return out


def save_keys(b: dict) -> dict:
    env_path = Path(".env")
    lines: dict[str, str] = {}
    if env_path.exists():
        for ln in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1)
                lines[k.strip()] = v.strip()
    mapping = {"gemini_key": "GEMINI_API_KEY", "groq_key": "GROQ_API_KEY",
               "openrouter_key": "OPENROUTER_API_KEY"}
    for src, dst in mapping.items():
        if b.get(src):
            lines[dst] = b[src]
            os.environ[dst] = b[src]
    env_path.write_text("\n".join(f"{k}={v}" for k, v in lines.items()) + "\n",
                        encoding="utf-8")
    # reader provider/model -> config.yaml (comment-preserving: simple key replace)
    if b.get("provider") or b.get("model"):
        from vvrag.config import load_config

        cfg = load_config("config.yaml")
        provider = b.get("provider") or cfg.reader.provider
        model = b.get("model") or cfg.reader.model
        text = Path("config.yaml").read_text(encoding="utf-8")
        text = text.replace(f"provider: \"{cfg.reader.provider}\"", f"provider: \"{provider}\"")
        text = text.replace(f"model: \"{cfg.reader.model}\"", f"model: \"{model}\"")
        Path("config.yaml").write_text(text, encoding="utf-8")
    return {"ok": True}


def do_upload(b: dict, index_dir: str) -> dict:
    from vvrag.ingest.run_ingest import run_ingest

    raw = Path("data/raw_uploads")
    raw.mkdir(parents=True, exist_ok=True)
    names = []
    for f in b.get("files", []):
        name = Path(f["name"]).name
        (raw / name).write_bytes(base64.b64decode(f["b64"]))
        names.append(name)
    if not names:
        raise ValueError("no files received")
    manifest = run_ingest(raw, index_dir, dpi=150, embedder="fake",
                          skip_ocr=bool(b.get("skip_ocr", True)),
                          qdrant_path=str(Path(index_dir) / "qdrant"))
    manifest["uploaded"] = names
    return manifest


def answer_ui(b: dict, index_dir: str, work_dir: str, default_provider: str) -> dict:
    from vvrag.ui.qa import answer_question

    if not b.get("question", "").strip():
        raise ValueError("empty question")
    return answer_question(
        b["question"], index_dir, work_dir,
        reader_mode="live" if b.get("reader_mode") == "live" else "mock",
        provider=b.get("provider") or default_provider,
        n_samples=int(b.get("n_samples", 3)),
        grounding=b.get("grounding", "span"),
        gate=b.get("gate", "none"),
        fixed_tau=float(b.get("fixed_tau", 0.5)))


def run(port: int = 8080, index_dir: str = INDEX_DIR):
    global INDEX_DIR
    INDEX_DIR = index_dir
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.index_dir = index_dir
    srv.work_dir = WORK_DIR
    print(f"altVRAG console at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    srv.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="altVRAG localhost console")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    args = ap.parse_args()
    run(args.port, args.index_dir)


if __name__ == "__main__":
    main()
