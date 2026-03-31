import os
   import sys
      import json
     import traceback
from typing import Any, Dict, Optional

sys.path.insert(0, "/app")

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
      from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from models import TriageAction, MedicationSafetyAction, SepsisManagementAction, ClinicalState
from environment import ClinicalTriageEnv, TASK_REGISTRY

app = FastAPI(title="ClinicalTriageEnv", version="1.0.0", docs_url="/docs")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_sessions: Dict[str, ClinicalTriageEnv] = {}
_default_env: Optional[ClinicalTriageEnv] = None

def get_default_env():
    global _default_env
    if _default_env is None:
        _default_env = ClinicalTriageEnv(task_id="triage_easy")
    return _default_env

class ResetRequest(BaseModel):
    task_id: Optional[str] = "triage_easy"

class GenericStepRequest(BaseModel):
    action: Dict[str, Any]

                         DASHBOARD_HTML = """<!DOCTYPE html>
                                         <html lang="en">
                                 <head>
                       <meta charset="UTF-8"/>
                          <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
                              <title>ClinicalTriageEnv — OpenEnv Healthcare AI</title>
                                         <link rel="preconnect" href="https://fonts.googleapis.com">
            <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #050d1a; --bg2: #081424; --bg3: #0c1e35; --panel: #0d1f38;
    --border: #1a3a5c; --border2: #0f2d4a;
    --cyan: #00d4ff; --cyan2: #00a8cc; --emerald: #00ff9d;
    --amber: #ffb800; --red: #ff4560; --purple: #a855f7;
    --text: #e2eeff; --text2: #7a9fc4; --text3: #3d6080;
    --mono: 'Space Mono', monospace; --sans: 'DM Sans', sans-serif; --display: 'Syne', sans-serif;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
                                                body { background:var(--bg); color:var(--text); font-family:var(--sans); min-height:100vh; overflow-x:hidden; }
  body::before { content:''; position:fixed; inset:0; background-image:linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px); background-size:40px 40px; pointer-events:none; z-index:0; }
  body::after { content:''; position:fixed; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--cyan),transparent); animation:scanline 3s ease-in-out infinite; z-index:100; }
         @keyframes scanline { 0%,100%{opacity:0.3;transform:scaleX(0.3);}50%{opacity:1;transform:scaleX(1);} }
                                                      header { position:relative; z-index:10; border-bottom:1px solid var(--border); background:rgba(5,13,26,0.95); backdrop-filter:blur(20px); padding:0 2rem; display:flex; align-items:center; justify-content:space-between; height:64px; animation:fadeDown 0.6s ease both; }
                                         @keyframes fadeDown { from{opacity:0;transform:translateY(-20px);}to{opacity:1;transform:translateY(0);} }
  .logo { display:flex; align-items:center; gap:12px; }
  .logo-icon { width:36px; height:36px; border:2px solid var(--cyan); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px; animation:pulse-border 2s ease infinite; }
                                                                @keyframes pulse-border { 0%,100%{box-shadow:0 0 0 0 rgba(0,212,255,0.4);}50%{box-shadow:0 0 0 8px rgba(0,212,255,0);} }
  .logo-text { font-family:var(--display); font-weight:800; font-size:1.1rem; letter-spacing:-0.02em; }
  .logo-text span { color:var(--cyan); }
  .header-status { display:flex; align-items:center; gap:12px; }
  .status-pill { display:flex; align-items:center; gap:6px; background:rgba(0,287,107,0.52); border:1px solid rgba(0,255,157,0.2); border-radius:20px; padding:4px 12px; font-family:var(--mono); font-size:0.7rem; color:var(--emerald); }
  .status-dot { width:6px; height:6px; border-radius:50%; background:var(--emerald); animation:blink 1.5s ease infinite; }
                                           @keyframes blink { 0%,100%{opacity:1;}50%{opacity:0.3;} }
  .badge { font-family:var(--mono); font-size:0.65rem; color:var(--text3); background:var(--bg3); border:1px solid var(--border); border-radius:4px; padding:3px 8px; }
  .hero { position:relative; z-index:1; padding:5rem 2rem 3rem; max-width:1200px; margin:0 auto; animation:fadeUp 0.8s ease 0.2s both; }
                        @keyframes fadeUp { from{opacity:0;transform:translateY(30px);}to{opacity:1;transform:translateY(0);} }
  .hero-tag { font-family:var(--mono); font-size:0.7rem; color:var(--cyan); letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px; }
  .hero-tag::before { content:''; width:24px; height:1px; background:var(--cyan); }
                                    .hero h1 { font-family:var(--display); font-size:clamp(3.6rem,7vw,8rem); font-weight:800; line-height:1.05; letter-spacing:-0.03em; margin-bottom:1.5rem; }
                                                                .hero h1 .line2 { color:transparent; -webkit-text-stroke:1px rgba(0,212,255,0.5); }
                                                                           .hero h1 .accent { color:var(--cyan); }
  .hero-desc { font-size:1.05rem; color:var(--text2); max-width:580px; line-height:1.7; margin-bottom:2.5rem; font-weight:300; }
  .hero-stats { display:flex; gap:2rem; flex-wrap:wrap; }
  .stat { display:flex; flex-direction:column; gap:2px; }
  .stat-val { font-family:var(--mono); font-size:1.6rem; font-weight:900; color:var(--cyan); line-height:1; }
  .stat-label { font-size:0.72rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.08em; }
  .stat-divider { width:1px; background:var(--border); align-self:stretch; }
  .main { position:relative; z-index:1; max-width:1200px; margin:0 auto; padding:0 2rem 4rem; display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }
                                                                @media(max-width:768px){.main{grid-template-columns:1fr;}}
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px; overflow:hidden; animation:fadeUp 0.8s ease both; transition:border-color 0.3s,box-shadow 0.3s; }
  .card:hover { border-color:rgba(0,212,255,0.3); box-shadow:0 0 30px rgba(0,212,255,0.05); }
  .card.full { grid-column:1/-1; }
  .card-header { padding:1rem 1.5rem; border-bottom:1px solid var(--border2); display:flex; align-items:center; justify-content:space-between; background:rgba(0,0,0,0.2); }
                               .card-title { font-family:var(--mono); font-size:0.72rem; color:var(--text2); text-transform:uppercase; letter-spacing:0.1em; display:flex; align-items:center; gap:8px; }
  .card-title::before { content:''; width:8px; height:8px; border-radius:50%; background:var(--cyan); box-shadow:0 0 8px var(--cyan); }
  .card-body { padding:1.5rem; }
  .tasks-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }
                               @media(max-width:600px){.tasks-grid{grid-template-columns:1fr;}}
  .task-card { background:var(--bg3); border:1px solid var(--border2); border-radius:8px; padding:1.2rem; cursor:pointer; transition:all 0.25s; position:relative; overflow:hidden; }
  .task-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--cyan),var(--emerald)); transform:scaleX(0); transition:transform 0.3s; transform-origin:left; }
  .task-card:hover::before { transform:scaleX(1); }
                        .task-card:hover { border-color:rgba(0,212,255,0.25); background:rgba(0,212,255,0.04); transform:translateY(-2px); }
  .task-type { font-family:var(--mono); font-size:0.6rem; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.5rem; }
  .task-name { font-family:var(--display); font-size:0.85rem; font-weight:700; color:var(--text); margin-bottom:0.4rem; line-height:1.3; }
                                     .task-desc { font-size:0.72rem; color:var(--text3); line-height:1.5; }
  .difficulty-badge { display:inline-flex; align-items:center; gap:4px; font-family:var(--mono); font-size:0.6rem; padding:2px 8px; border-radius:3px; margin-top:0.8rem; text-transform:uppercase; letter-spacing:0.08em; }
  .diff-easy { background:rgba(0,255,157,0.1); color:var(--emerald); border:1px solid rgba(0,255,157,0.2); }
  .diff-medium { background:rgba(255,184,0,0.1); color:var(--amber); border:1px solid rgba(255,184,0,0.2); }
  .diff-hard { background:rgba(255,69,96,0.1); color:var(--red); border:1px solid rgba(255,69,96,0.2); }
                                           .api-section { display:flex; flex-direction:column; gap:0.8rem; }
  .endpoint-row { display:flex; align-items:center; gap:10px; padding:0.8rem 1rem; background:var(--bg3); border:1px solid var(--border2); border-radius:6px; cursor:pointer; transition:all 0.2s; text-decoration:none; }
  .endpoint-row:hover { border-color:rgba(0,212,255,0.3); background:rgba(0,212,255,0.04); }
  .method { font-family:var(--mono); font-size:0.65rem; font-weight:700; padding:2px 6px; border-radius:3px; min-width:36px; text-align:center; }
  .get { background:rgba(0,255,157,0.15); color:var(--emerald); }
  .post { background:rgba(0,212,255,0.15); color:var(--cyan); }
  .ws { background:rgba(168,85,247,0.15); color:var(--purple); }
  .endpoint-path { font-family:var(--mono); font-size:0.78rem; color:var(--text); flex:1; }
  .endpoint-desc { font-size:0.72rem; color:var(--text3); }
  .demo-section { display:flex; flex-direction:column; gap:1rem; }
  select { width:100%; background:var(--bg); border:1px solid var(--border); border-radius:6px; color:var(--text); font-family:var(--mono); font-size:0.78rem; padding:0.7rem 1rem; outline:none; transition:border-color 0.2s; }
  select:focus { border-color:var(--cyan); }
                                      .btn { display:inline-flex; align-items:center; gap:8px; padding:0.7rem 1.4rem; border-radius:6px; font-family:var(--mono); font-size:0.75rem; font-weight:700; letter-spacing:0.05em; cursor:pointer; border:none; transition:all 0.2s; text-transform:uppercase; }
                .btn-primary { background:var(--cyan); color:var(--bg); }
                                     .btn-primary:hover { background:#00eeff; box-shadow:0 0 20px rgba(0,212,255,0.4); transform:translateY(-1px); }
  .btn-secondary { background:transparent; color:var(--cyan); border:1px solid var(--cyan); }
  .btn-secondary:hover { background:rgba(0,212,255,0.08); }
  .btn-row { display:flex; gap:8px; flex-wrap:wrap; }
  .response-box { background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:1rem; font-family:var(--mono); font-size:0.72rem; color:var(--emerald); min-height:100px; max-height:280px; overflow-y:auto; white-space:pre-wrap; word-break:break-all; line-height:1.7; }
  .response-label { font-family:var(--mono); font-size:0.65rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; }
  .arch { display:flex; align-items:center; justify-content:center; gap:0; flex-wrap:wrap; padding:1rem 0; }
  .arch-box { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:0.8rem 1.2rem; text-align:center; min-width:100px; }
  .arch-icon { font-size:1.4rem; margin-bottom:4px; }
  .arch-label { font-family:var(--mono); font-size:0.65rem; color:var(--text2); text-transform:uppercase; letter-spacing:0.08em; }
  .arch-arrow { font-family:var(--mono); color:var(--cyan); font-size:1rem; padding:0 0.5rem; opacity:0.6; }
  footer { position:relative; z-index:1; border-top:1px solid var(--border); padding:1.5rem 2rem; display:flex; align-items:center; justify-content:space-between; max-width:1200px; margin:0 auto; }
  .footer-text { font-family:var(--mono); font-size:0.68rem; color:var(--text3); }
  .footer-links { display:flex; gap:1rem; }
  .footer-links a { font-family:var(--mono); font-size:0.68rem; color:var(--text3); text-decoration:none; transition:color 0.2s; }
  .footer-links a:hover { color:var(--cyan); }
  @keyframes glow-pulse { 0%,100%{text-shadow:0 0 8px currentColor;}50%{text-shadow:0 0 20px currentColor,0 0 40px currentColor;} }
  .glow { animation:glow-pulse 3s ease infinite; }
  .card:nth-child(1){animation-delay:0.3s;} .card:nth-child(2){animation-delay:0.8s;} .card:nth-child(3){animation-delay:0.5s;} .card:nth-child(4){animation-delay:0.6s;}
  ::-webkit-scrollbar{width:4px;} ::-webkit-scrollbar-track{background:var(--bg);} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">🏥</div>
    <div class="logo-text">Clinical<span>Triage</span>Env</div>
  </div>
  <div class="header-status">
    <div class="status-pill"><div class="status-dot"></div>LIVE</div>
    <div class="badge">OpenEnv v0.1</div>
    <div class="badge">v1.0.0</div>
  </div>
</header>
<div class="hero">
  <div class="hero-tag">OpenEnv · Healthcare AI · Reinforcement Learning</div>
  <h1>Clinical Decision<br><span class="line2">Intelligence</span> <span class="accent">Engine</span></h1>
  <p class="hero-desc">A real-world AI training environment where agents learn to make life-or-death clinical decisions — emergency triage, drug safety, sepsis management.</p>
  <div class="hero-stats">
    <div class="stat"><div class="stat-val glow">9</div><div class="stat-label">Clinical Tasks</div></div>
    <div class="stat-divider"></div>
    <div class="stat"><div class="stat-val glow">3</div><div class="stat-label">Task Domains</div></div>
    <div class="stat-divider"></div>
    <div class="stat"><div class="stat-val glow">0→1</div><div class="stat-label">Partial Scoring</div></div>
    <div class="stat-divider"></div>
    <div class="stat"><div class="stat-val glow">0.58</div><div class="stat-label">Baseline Score</div></div>
  </div>
</div>
<div class="main">
  <div class="card full" style="animation-delay:0.3s">
    <div class="card-header">
      <div class="card-title">Task Registry — 9 Clinical Tasks</div>
      <div class="badge">Easy → Medium → Hard</div>
    </div>
    <div class="card-body">
      <div class="tasks-grid" id="tasks-grid"><div style="color:var(--text3);font-family:var(--mono);font-size:0.75rem;">Loading tasks...</div></div>
    </div>
  </div>
  <div class="card" style="animation-delay:0.4s">
    <div class="card-header"><div class="card-title">Live Environment Demo</div><div class="badge">Interactive</div></div>
    <div class="card-body demo-section">
      <div>
        <div class="response-label">Select Task</div>
        <select id="task-select">
          <option value="triage_easy">🟢 Triage — Easy</option>
          <option value="triage_medium">🟡 Triage — Medium (ACS)</option>
          <option value="triage_hard">🔴 Triage — Hard (Stroke)</option>
          <option value="med_safety_easy">🟢 Med Safety — Easy</option>
          <option value="med_safety_medium">🟡 Med Safety — Medium</option>
          <option value="med_safety_hard">🔴 Med Safety — Hard</option>
          <option value="sepsis_easy">🟢 Sepsis — Easy</option>
          <option value="sepsis_medium">🟡 Sepsis — Medium</option>
          <option value="sepsis_hard">🔴 Sepsis — Hard</option>
        </select>
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" onclick="doReset()">⚡ Reset Episode</button>
        <button class="btn btn-secondary" onclick="doHealth()">❤ Health Check</button>
      </div>
      <div>
        <div class="response-label">Response</div>
        <div class="response-box" id="response-box">// Press Reset Episode to start a clinical scenario...</div>
      </div>
    </div>
  </div>
  <div class="card" style="animation-delay:0.5s">
    <div class="card-header"><div class="card-title">API Endpoints</div><div class="badge">OpenEnv Spec</div></div>
    <div class="card-body api-section">
      <a class="endpoint-row" href="/health" target="_blank"><span class="method get">GET</span><span class="endpoint-path">/health</span><span class="endpoint-desc">Ping</span></a>
      <a class="endpoint-row" href="/tasks" target="_blank"><span class="method get">GET</span><span class="endpoint-path">/tasks</span><span class="endpoint-desc">List all tasks</span></a>
      <div class="endpoint-row" onclick="doReset()"><span class="method post">POST</span><span class="endpoint-path">/reset</span><span class="endpoint-desc">Start episode</span></div>
      <div class="endpoint-row"><span class="method post">POST</span><span class="endpoint-path">/step</span><span class="endpoint-desc">Submit action</span></div>
      <a class="endpoint-row" href="/state" target="_blank"><span class="method get">GET</span><span class="endpoint-path">/state</span><span class="endpoint-desc">Episode state</span></a>
      <div class="endpoint-row"><span class="method ws">WS</span><span class="endpoint-path">/ws</span><span class="endpoint-desc">WebSocket stream</span></div>
      <a class="endpoint-row" href="/docs" target="_blank"><span class="method get">GET</span><span class="endpoint-path">/docs</span><span class="endpoint-desc">Swagger UI ↗</span></a>
    </div>
  </div>
  <div class="card full" style="animation-delay:0.6s">
    <div class="card-header"><div class="card-title">Environment Architecture</div><div class="badge">OpenEnv Compatible</div></div>
    <div class="card-body">
      <div class="arch">
        <div class="arch-box"><div class="arch-icon">🤖</div><div class="arch-label">AI Agent</div></div>
        <div class="arch-arrow">→ action →</div>
        <div class="arch-box"><div class="arch-icon">⚡</div><div class="arch-label">step()</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-box"><div class="arch-icon">🏥</div><div class="arch-label">Grader</div></div>
        <div class="arch-arrow">→ reward →</div>
        <div class="arch-box"><div class="arch-icon">📊</div><div class="arch-label">Observation</div></div>
        <div class="arch-arrow">→</div>
        <div class="arch-box"><div class="arch-icon">🤖</div><div class="arch-label">AI Agent</div></div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.5rem;">
        <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:1rem;">
          <div style="font-family:var(--mono);font-size:0.65rem;color:var(--cyan);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">🚨 ED Triage</div>
          <div style="font-size:0.78rem;color:var(--text2);line-height:1.6;">ESI 1–5 assignment with undertriage penalties. Acute stroke, ACS, sepsis recognition.</div>
        </div>
        <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:1rem;">
          <div style="font-family:var(--mono);font-size:0.65rem;color:var(--amber);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">💊 Med Safety</div>
          <div style="font-size:0.78rem;color:var(--text2);line-height:1.6;">Drug interaction detection, contraindications, CYP450 pharmacokinetics, rhabdomyolysis.</div>
        </div>
        <div style="background:var(--bg3);border:1px solid var(--border2);border-radius:8px;padding:1rem;">
          <div style="font-family:var(--mono);font-size:0.65rem;color:var(--red);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">🔴 Sepsis Bundle</div>
          <div style="font-size:0.78rem;color:var(--text2);line-height:1.6;">Hour-1 SSC bundle execution. Vasopressor decisions, allergy-safe antibiotics, source control.</div>
        </div>
      </div>
    </div>
  </div>
</div>
<footer>
  <div class="footer-text">ClinicalTriageEnv · OpenEnv Hackathon 2025 · MIT License</div>
  <div class="footer-links">
    <a href="/docs">API Docs</a>
    <a href="/tasks">Tasks</a>
    <a href="/health">Health</a>
  </div>
</footer>
<script>
const BASE = window.location.origin;
const TYPE_COLORS = {triage:'var(--cyan)',medication_safety:'var(--amber)',sepsis:'var(--red)'};
async function loadTasks() {
  try {
    const r = await fetch(BASE+'/tasks');
    const data = await r.json();
    const grid = document.getElementById('tasks-grid');
    grid.innerHTML = '';
    Object.entries(data.tasks).forEach(([id,t]) => {
      const color = TYPE_COLORS[t.type]||'var(--cyan)';
      const diffClass = 'diff-'+t.difficulty;
      grid.innerHTML += `<div class="task-card" onclick="document.getElementById('task-select').value='${id}';doReset();">
        <div class="task-type" style="color:${color}">${t.type.replace('_',' ')}</div>
        <div class="task-name">${t.name}</div>
        <div class="task-desc">${t.description}</div>
        <span class="difficulty-badge ${diffClass}">${t.difficulty}</span>
      </div>`;
    });
  } catch(e) {
    document.getElementById('tasks-grid').innerHTML = '<div style="color:var(--red);font-family:var(--mono);font-size:0.75rem;">Failed to load tasks</div>';
  }
}
function setResponse(text) {
  document.getElementById('response-box').textContent = text;
}
async function doHealth() {
  setResponse('Pinging /health...');
  try {
    const r = await fetch(BASE+'/health');
    const d = await r.json();
    setResponse(JSON.stringify(d,null,2));
  } catch(e) { setResponse('Error: '+e.message); }
}
async function doReset() {
  const taskId = document.getElementById('task-select').value;
  setResponse('Resetting → '+taskId+'...');
  try {
    const r = await fetch(BASE+'/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId})});
    const d = await r.json();
    const obs = d.observation;
    const p = obs.patient;
    const v = p.vitals;
    setResponse([
      '✅ Episode started — '+d.task_name,
      '📋 Difficulty: '+d.difficulty.toUpperCase(),
      '',
      '👤 PATIENT: '+p.patient_id+' | Age '+p.age+' '+p.sex,
      '🩺 Chief Complaint: '+p.chief_complaint,
      '',
      '📊 VITALS:',
      '   HR: '+v.heart_rate+' bpm  |  BP: '+v.systolic_bp+'/'+v.diastolic_bp+' mmHg',
      '   Temp: '+v.temperature+'°C  |  SpO2: '+v.spo2+'%',
      '   RR: '+v.respiratory_rate+'/min  |  GCS: '+v.glasgow_coma_scale+'/15',
      '',
      '💊 Medications: '+(p.current_medications.length>0?p.current_medications.map(m=>m.name).join(', '):'None'),
      '⚠️  Allergies: '+(p.allergies.length>0?p.allergies.join(', '):'NKDA'),
      '',
      '📝 TASK:',
      obs.task_description,
      '',
      '— Use POST /step with your action to respond —'
    ].join('\\n'));
  } catch(e) { setResponse('Error: '+e.message); }
}
loadTasks();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=DASHBOARD_HTML)

@app.get("/health")
def health():
    return {"status": "ok", "environment": "ClinicalTriageEnv"}

@app.get("/tasks")
def list_tasks():
    return {"tasks": ClinicalTriageEnv.list_tasks()}

@app.post("/reset")
def reset(req: ResetRequest):
    global _default_env
    task_id = req.task_id or "triage_easy"
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    _default_env = ClinicalTriageEnv(task_id=task_id)
    obs = _default_env.reset()
    return {"observation": obs.model_dump(), "task_id": task_id, "task_name": TASK_REGISTRY[task_id]["name"], "difficulty": TASK_REGISTRY[task_id]["difficulty"]}

@app.post("/step")
def step(action_body: GenericStepRequest):
    env = get_default_env()
    task_type = env.task_meta["type"]
    action_data = action_body.action
    try:
        if task_type == "triage":
            action = TriageAction(**action_data)
        elif task_type == "medication_safety":
            action = MedicationSafetyAction(**action_data)
        elif task_type == "sepsis":
            action = SepsisManagementAction(**action_data)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

@app.get("/state")
def get_state():
    return get_default_env().state().model_dump()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    session_env = ClinicalTriageEnv(task_id="triage_easy")
    _sessions[session_id] = session_env
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type","")
            if msg_type == "reset":
                task_id = msg.get("task_id","triage_easy")
                if task_id not in TASK_REGISTRY:
                    await websocket.send_text(json.dumps({"type":"error","message":f"Unknown task_id: {task_id}"}))
                    continue
                session_env = ClinicalTriageEnv(task_id=task_id)
                _sessions[session_id] = session_env
                obs = session_env.reset()
                await websocket.send_text(json.dumps({"type":"observation","observation":obs.model_dump(),"task_id":task_id}))
            elif msg_type == "step":
                action_data = msg.get("action",{})
                task_type = session_env.task_meta["type"]
                try:
                    if task_type == "triage":
                        action = TriageAction(**action_data)
                    elif task_type == "medication_safety":
                        action = MedicationSafetyAction(**action_data)
                    else:
                        action = SepsisManagementAction(**action_data)
                    obs, reward, done, info = session_env.step(action)
                    await websocket.send_text(json.dumps({"type":"step_result","observation":obs.model_dump(),"reward":reward,"done":done,"info":info}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type":"error","message":str(e)}))
            elif msg_type == "state":
                await websocket.send_text(json.dumps({"type":"state","state":session_env.state().model_dump()}))
    except WebSocketDisconnect:
        _sessions.pop(session_id, None)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    print(f"Starting on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
