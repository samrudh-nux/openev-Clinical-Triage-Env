---
title: ClinicalTriageEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: true
license: mit
tags:
  - healthcare
  - clinical-decision-support
  - triage
  - medication-safety
  - sepsis
  - openenv
  - rl-training
  - agentic-ai
  - llm-evaluation
  - reinforcement-learning
---

<div align="center">

# 🏥 ClinicalTriageEnv

### *A High-Fidelity Healthcare AI Training & Evaluation Environment*

**Built for the Meta × Scaler Open-Env Hackathon**

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v0.1%20Compliant-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/samrudh-nux/ClinicalTriageEnv)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HF%20Space-Live-ff9f00?style=for-the-badge)](https://huggingface.co/spaces/samrudh-nux/ClinicalTriageEnv) 
     
#Click live to watch demo.

---

> **ClinicalTriageEnv** is a fully OpenEnv-spec-compliant environment that stress-tests AI agents on *real-world clinical decision-making* — the kind where a wrong answer harms a human being.
>
> It covers three high-stakes medical domains across nine difficulty-tiered tasks, backed by programmatic multi-component graders and a shaped reward system designed to surface the true limits of any LLM.

---

## Author : Samrudh

---

</div>

## 📋 Table of Contents

- [Why Clinical AI?](#-why-clinical-ai)
- [Environment Overview](#-environment-overview)
- [Task Catalogue](#-task-catalogue-9-tasks-3-domains-3-difficulties)
- [OpenEnv Spec Compliance](#-openenv-spec-compliance)
- [Architecture & Codebase](#-architecture--codebase)
- [Grading System](#-grading-system-in-depth)
- [Reward Engineering](#-reward-engineering)
- [Baseline Results](#-baseline-results-meta-llama-3370b)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Running the Benchmark](#-running-the-inference-benchmark)
- [Hackathon Alignment](#-meta--scaler-open-env-hackathon-alignment)
- [Future Roadmap](#-future-roadmap)

---

## 🌍 Why Clinical AI?

Emergency departments worldwide face crushing patient loads. Medication errors kill an estimated **3-million people** anually across globe. Sepsis, when missed or delayed, carries a mortality rate exceeding 30%.

AI agents capable of safe, accurate clinical reasoning could:
- **Reduce undertriage errors** that send critical patients to waiting rooms
- **Flag life-threatening drug interactions** before a pharmacist can
- **Execute sepsis bundles** in under 60 minutes, slashing mortality

But testing these agents requires more than a Q&A dataset. It requires an *environment* — one that presents real patient scenarios, enforces clinical protocols, penalizes patient-safety failures, and produces granular, interpretable feedback. That is exactly what **ClinicalTriageEnv** provides.

---

## 🏗️ Environment Overview

ClinicalTriageEnv implements the **OpenEnv specification** — the standard interface for training and evaluating agentic LLMs in structured environments.

```
reset(task_id)  →  Observation
step(action)    →  (Observation, reward: float, done: bool, info: dict)
state()         →  ClinicalState
```

### Three Clinical Domains

| Domain | Real-World Analog | Protocol |
|--------|------------------|----------|
| 🚨 **Emergency Triage** | ED nurse/physician triage decision | ESI v4 (Emergency Severity Index) |
| 💊 **Medication Safety** | Clinical pharmacist review | FDA/CYP450 interaction databases |
| 🦠 **Sepsis Management** | Intensivist Hour-1 bundle | SSC 2021 Surviving Sepsis Campaign |

### Key Design Principles

- **Medically accurate scenarios** — all patient cases reflect real clinical presentations validated against evidence-based guidelines
- **Programmatic grading** — no LLM judge; pure rule-based + NLP-based graders that are deterministic and auditable
- **Partial credit** — component-wise scoring rewards partial clinical reasoning, not just correct final answers
- **Patient safety enforcement** — critical errors (undertriage, allergy violations, missed sepsis criteria) trigger hard penalties regardless of other scores
- **Difficulty gradient** — Easy → Medium → Hard within each domain, enabling curriculum learning research

---

## 📋 Task Catalogue: 9 Tasks · 3 Domains · 3 Difficulties

### 🚨 Domain 1: Emergency Department Triage (ESI Level Assignment)

The Emergency Severity Index (ESI) is the global standard for ED patient prioritization. Assigning the wrong ESI level — especially **undertriage** (sending a critical patient to wait) — is a life-threatening error. The environment penalizes undertriage asymmetrically.

| Task ID | Difficulty | Clinical Scenario | Ground Truth ESI |
|---------|-----------|-------------------|-----------------|
| `triage_easy` | 🟢 Easy | Non-urgent presentation; stable vitals | ESI-4 or 5 |
| `triage_medium` | 🟡 Medium | Potential ACS (Acute Coronary Syndrome); high-risk chest pain | ESI-2 |
| `triage_hard` | 🔴 Hard | Acute stroke patient on anticoagulation; time-critical neurological deficit | ESI-1 or 2 |

**What the agent must do:**
1. Assign an ESI level (1–5)
2. Provide clinical rationale citing specific vital signs and symptoms
3. List immediate interventions (e.g., `12-lead ECG`, `IV access`, `activate stroke team`)

---

### 💊 Domain 2: Medication Safety Review

Clinical pharmacology at its most dangerous. The agent reviews a patient's full medication regimen and flags interactions, contraindications, and dosing errors — each with severity classification and actionable recommendations.

| Task ID | Difficulty | Clinical Scenario | Key Hazard |
|---------|-----------|-------------------|------------|
| `med_safety_easy` | 🟢 Easy | Clean medication list; straightforward interactions | Minor interactions only |
| `med_safety_medium` | 🟡 Medium | Post-MI on triple antithrombotic therapy (aspirin + clopidogrel + warfarin); CKD + diabetes | Major bleed risk; metformin contraindication |
| `med_safety_hard` | 🔴 Hard | HIV patient on ritonavir presenting with rhabdomyolysis | Simvastatin + ritonavir = life-threatening CYP3A4 interaction |

**What the agent must do:**
1. Flag all drug-drug interactions with mechanism
2. Identify contraindications given the patient's comorbidities
3. Detect dosing errors
4. Classify overall severity (`safe` → `minor` → `moderate` → `major` → `critical`)
5. Provide specific recommended changes (discontinue, switch, reduce, monitor)

The hard scenario specifically tests knowledge of **CYP3A4 inhibition** — ritonavir boosting simvastatin levels by up to 3000%, directly causing rhabdomyolysis. Missing this = critical failure.

---

### 🦠 Domain 3: Sepsis Recognition & Management

Sepsis is a medical emergency where every minute of delayed antibiotic therapy increases mortality by ~7%. The **SSC Hour-1 Bundle** (Surviving Sepsis Campaign) is the gold standard protocol. The environment checks every bundle element.

| Task ID | Difficulty | Clinical Scenario | Key Challenge |
|---------|-----------|-------------------|---------------|
| `sepsis_easy` | 🟢 Easy | Urosepsis with documented penicillin allergy | Allergy-appropriate antibiotic selection |
| `sepsis_medium` | 🟡 Medium | Septic shock in elderly nursing home patient; MRSA history; vasopressor needed | Vasopressor decision-making (MAP < 65) |
| `sepsis_hard` | 🔴 Hard | Post-operative anastomotic leak; multi-organ failure; DIC; vancomycin allergy | Complex antibiotic selection + source control identification |

**What the agent must do (Hour-1 SSC Bundle):**
- [ ] Blood cultures × 2 **before** antibiotics
- [ ] Broad-spectrum antibiotics (allergy-checked!)
- [ ] Serum lactate measurement
- [ ] 30 mL/kg IV crystalloid if MAP < 65 or lactate ≥ 4
- [ ] Vasopressors (norepinephrine first-line) if MAP < 65 despite fluids

---

## ✅ OpenEnv Spec Compliance

ClinicalTriageEnv is fully compliant with the **OpenEnv v0.1 specification**.

```yaml
# openenv.yaml (excerpt)
openenv_spec: "0.1"

action_space:
  type: pydantic
  classes:
    - TriageAction
    - MedicationSafetyAction
    - SepsisManagementAction

observation_space:
  type: pydantic
  classes:
    - TriageObservation
    - MedicationSafetyObservation
    - SepsisManagementObservation

reward:
  type: continuous
  range: [-1.0, 1.5]

episode:
  max_steps: 3
  terminates_on_action: true
  reset_required: true
```

### HTTP API Endpoints

The environment exposes a **FastAPI** server:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode (optionally switch task) |
| `POST` | `/step` | Submit an action; receive reward + observation |
| `GET` | `/state` | Inspect current episode state |
| `GET` | `/tasks` | List all 9 available tasks with metadata |
| `GET` | `/health` | Server health check |
| `WS` | `/ws` | WebSocket for streaming interaction |

---

## 🏛️ Architecture & Codebase

```
ClinicalTriageEnv/
├── environment.py      # Core OpenEnv class: reset(), step(), state()
├── models.py           # Pydantic schemas: Actions, Observations, PatientRecord
├── graders.py          # Programmatic graders (3 domain-specific graders)
├── scenarios.py        # Patient scenario library (9 medically accurate cases)
├── inference.py        # LLM inference engine + benchmark runner
├── ml_engine.py        # ML utilities and supporting functions
├── app.py              # FastAPI server (REST + WebSocket)
├── index.html          # Interactive web UI for live testing
├── openenv.yaml        # OpenEnv spec manifest
├── Dockerfile          # Docker deployment (python:3.11-slim, port 7860)
└── requirements.txt    # Dependencies
```

### Data Flow

```
User/Agent
    │
    ▼
POST /step  {action: TriageAction}
    │
    ▼
ClinicalTriageEnv.step(action)
    ├── _grade_action(action, task_type)
    │       └── TriageGrader / MedicationSafetyGrader / SepsisGrader
    │               └── 7–9 component scores → weighted final score
    │
    ├── _compute_step_reward(grade_result)
    │       └── base_reward - safety_penalty + efficiency_bonus
    │               × difficulty_multiplier
    │
    └── _build_observation(done, reward, feedback)
            └── Domain-specific Observation with patient + qSOFA / drug_info
    │
    ▼
(Observation, float, bool, dict)
```

### `models.py` — Pydantic Schema Design

Every patient is represented as a structured `PatientRecord`:

```python
class PatientRecord(BaseModel):
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    vitals: VitalSigns          # HR, BP, SpO2, Temp, RR, GCS
    symptoms: List[str]
    medical_history: List[str]
    current_medications: List[Medication]
    lab_results: Dict[str, Any]
    arrival_time_minutes: int
    allergies: List[str]
```

Actions are strongly typed with field-level validation:

```python
class TriageAction(Action):
    esi_level: int              # Field(ge=1, le=5)
    rationale: str              # Field(min_length=10)
    recommended_immediate_interventions: List[str]

class SepsisManagementAction(Action):
    sepsis_diagnosis: str       # "sepsis" | "septic_shock" | "SIRS_only" | "no_sepsis"
    blood_cultures_ordered: bool
    antibiotics_ordered: bool
    antibiotic_choice: Optional[str]
    lactate_ordered: bool
    iv_fluid_bolus_ml: int
    vasopressor_ordered: bool
    vasopressor_choice: Optional[str]
    source_control_identified: Optional[str]
    clinical_rationale: str
    time_to_antibiotics_minutes: Optional[int]
```

---

## 📊 Grading System In-Depth

All three graders are **fully programmatic** — no LLM judges, no human annotation required. Each grader decomposes the clinical decision into multiple independent axes and assigns partial credit.

### Grader 1: `TriageGrader` — 7 Components

```
Component                    Weight   Description
──────────────────────────   ──────   ─────────────────────────────────────────
esi_accuracy                 0.40     Exact ESI match: 1.0; ±1: 0.55; ±2: 0.20
acceptable_range_bonus       0.05     Bonus if ESI within clinical tolerance range
rationale_keywords           0.15     NLP keyword density (vitals, urgency terms)
clinical_flags               0.10     Specific abnormal vital sign values cited
intervention_recall          0.15     Fuzzy recall of expected interventions
intervention_precision       0.05     Penalty for hallucinated interventions
time_sensitivity             0.10     Urgency language for ESI 1–3 patients
```

**Critical safety rules hardcoded:**
- ESI-1 patient assigned ESI-3+: `-0.45` undertriage penalty + critical error flag
- ESI-1 patient assigned ESI-4/5: score capped at `0.0` → automatic fail
- ESI-1 assigned ESI-2: `-0.20` minor undertriage penalty

### Grader 2: `MedicationSafetyGrader` — 7 Components

```
Component                    Weight   Description
──────────────────────────   ──────   ─────────────────────────────────────────
interaction_recall           0.28     Fuzzy recall of GT drug-drug interactions
contraindication_recall      0.18     Recall of GT drug-condition contraindications
dosing_error_recall          0.10     Recall of GT dosing errors
severity_accuracy            0.15     Severity classification (safe→critical scale)
recommended_changes          0.12     Actionable verbs + correct drug references
rationale_depth              0.12     Length + CYP450/pharmacokinetic keyword density
fp_penalty                   0.05     Penalty for hallucinated interactions
```

**Critical safety rules:**
- GT=critical, proposed=safe: score hard-capped at `0.20`
- GT=critical, proposed=minor: score hard-capped at `0.35`
- GT=major, proposed=safe: critical error flagged

### Grader 3: `SepsisGrader` — 9 Components

```
Component                    Weight   Description
──────────────────────────   ──────   ─────────────────────────────────────────
diagnosis_accuracy           0.15     Spectrum: SIRS→sepsis→septic_shock
blood_cultures               0.08     Boolean: cultures drawn before antibiotics
antibiotics                  0.15     Boolean: broad-spectrum antibiotics ordered
antibiotic_safety            0.12     Allergy cross-check; allergy violation = 0.0
lactate                      0.08     Boolean: serum lactate ordered
fluid_volume                 0.10     Partial credit for 30mL/kg target (±20%)
vasopressor                  0.12     NE=1.0, vasopressin=0.75, phenylephrine=0.30
source_control               0.05     Infection source identification (fuzzy match)
rationale_depth              0.15     SSC keyword density + length scoring
```

**Bonus:** `time_to_antibiotics_minutes ≤ 30` → +0.03 reward; `> 120` → -0.02

### NLP Grading Utilities

All three graders share reusable NLP primitives from `graders.py`:

```python
def _fuzzy_list_recall(proposed, ground_truth) -> float:
    """Word-level fuzzy match: ≥40% word overlap = item detected."""

def _false_positive_rate(proposed, ground_truth) -> float:
    """Fraction of proposed items with no GT support → hallucination penalty."""

def _keyword_score(text, keywords, threshold) -> float:
    """Fraction of expected clinical keywords found in text."""

def _token_overlap(a, b) -> float:
    """Jaccard similarity between token sets of two strings."""
```

---

## 🎯 Reward Engineering

The reward function is carefully shaped to:
1. Provide signal throughout the episode (not just at the end)
2. Penalize patient safety failures heavily
3. Reward efficient, accurate clinical reasoning

```python
def _compute_step_reward(grade_result, step_num, max_steps) -> float:
    difficulty_multiplier = {"easy": 0.8, "medium": 1.0, "hard": 1.3}[difficulty]
    
    base_reward    = grade_result.score                          # 0.0 – 1.0
    safety_penalty = 0.3 × len(grade_result.critical_errors)    # Per safety violation
    efficiency_bonus = 0.05 × max(0, max_steps - step_num)      # Speed reward
    
    raw = (base_reward - safety_penalty + efficiency_bonus) × difficulty_multiplier
    return clamp(raw, -1.0, 1.5)
```

**Reward range:** `[-1.0, 1.5]`
- `-1.0`: Critical safety failure (e.g., gave contraindicated drug to allergic patient)
- `0.0`: Minimum acceptable (passing threshold: `0.60`)
- `1.0`: Near-perfect clinical response
- `1.5`: Perfect response on hardest difficulty with speed bonus

---

## 📈 Baseline Results: Meta-Llama 3.3-70B

Baseline inference was run using `inference.py` with `meta-llama/Llama-3.3-70B-Instruct` via the HuggingFace Router (Chain-of-Thought enabled).

| Task | Difficulty | Score | Status |
|------|-----------|-------|--------|
| `triage_easy` | 🟢 Easy | **0.78** | ✅ Pass |
| `triage_medium` | 🟡 Medium | **0.62** | ✅ Pass |
| `triage_hard` | 🔴 Hard | **0.45** | ❌ Fail |
| `med_safety_easy` | 🟢 Easy | **0.90** | ✅ Pass |
| `med_safety_medium` | 🟡 Medium | **0.58** | ❌ Fail |
| `med_safety_hard` | 🔴 Hard | **0.31** | ❌ Fail |
| `sepsis_easy` | 🟢 Easy | **0.72** | ✅ Pass |
| `sepsis_medium` | 🟡 Medium | **0.55** | ❌ Fail |
| `sepsis_hard` | 🔴 Hard | **0.28** | ❌ Fail |
| **Overall Mean** | — | **0.58** | — |

**Key findings:**
- Even the strongest open-weight model (70B) fails on hard medication safety, indicating the real difficulty of CYP450 interaction reasoning
- Sepsis hard is the lowest-scoring task — multi-organ failure + rare antibiotic combinations stress-test rare clinical knowledge
- The baseline of **0.58 mean** provides a clear target for future fine-tuned or RLHF-trained agents to beat

---

## 🚀 Quick Start

### Option 1: Use the Live Space

Visit the running HuggingFace Space and interact with the environment directly via the **web UI**:

👉 **[https://huggingface.co/spaces/samrudh-nux/ClinicalTriageEnv](https://huggingface.co/spaces/samrudh-nux/ClinicalTriageEnv)**

### Option 2: Python Client

```python
from environment import ClinicalTriageEnv
from models import TriageAction

# Initialize with any of 9 task IDs
env = ClinicalTriageEnv(task_id="triage_medium")
obs = env.reset()

print(f"Patient: {obs.patient.age}yo {obs.patient.sex}")
print(f"Chief Complaint: {obs.patient.chief_complaint}")
print(f"Vitals: HR={obs.patient.vitals.heart_rate}, "
      f"BP={obs.patient.vitals.systolic_bp}/{obs.patient.vitals.diastolic_bp}")

# Submit a clinical decision
action = TriageAction(
    esi_level=2,
    rationale=(
        "58yo male with crushing chest pain, diaphoresis, and ST elevation. "
        "HR 110 bpm, BP 88/60 mmHg — hemodynamically unstable. "
        "High suspicion for STEMI. Requires immediate ESI-2 with STEMI activation."
    ),
    recommended_immediate_interventions=[
        "12-lead ECG", "IV access x2", "aspirin 325mg",
        "activate cath lab", "continuous cardiac monitoring", "troponin/CK-MB"
    ]
)

obs, reward, done, info = env.step(action)

print(f"Reward:           {reward:.3f}")
print(f"Grade:            {info['grade']:.3f}")
print(f"Passed:           {info['passed']}")
print(f"Component Scores: {info['component_scores']}")
if info['critical_errors']:
    print(f"Safety Alerts:    {info['critical_errors']}")
```

### Option 3: REST API

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "sepsis_medium"}'

# Submit an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "sepsis_diagnosis": "septic_shock",
    "blood_cultures_ordered": true,
    "antibiotics_ordered": true,
    "antibiotic_choice": "vancomycin_piperacillin_tazobactam",
    "lactate_ordered": true,
    "iv_fluid_bolus_ml": 2100,
    "vasopressor_ordered": true,
    "vasopressor_choice": "norepinephrine",
    "source_control_identified": "urinary tract",
    "clinical_rationale": "Patient meets septic shock criteria: MAP<65, lactate elevated...",
    "time_to_antibiotics_minutes": 30
  }'
```

### Option 4: Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  clinical-triage-env
```

---

## 📖 API Reference

### List All Tasks

```
GET /tasks
```

```json
{
  "triage_easy": {
    "name": "Emergency Triage - Easy",
    "type": "triage",
    "difficulty": "easy",
    "description": "Assign ESI level to non-urgent patient..."
  },
  "med_safety_hard": {
    "name": "Medication Safety Review - Hard",
    "type": "medication_safety",
    "difficulty": "hard",
    "description": "HIV PI + statin rhabdomyolysis. Life-threatening CYP3A4..."
  }
}
```

### Step Response Schema

```json
{
  "observation": {
    "patient": { "...PatientRecord..." },
    "task_description": "string",
    "current_step": 1,
    "max_steps": 3,
    "feedback": "=== TRIAGE GRADER FEEDBACK ===\n...",
    "score_so_far": 0.84,
    "done": true,
    "reward": 0.84
  },
  "reward": 0.84,
  "done": true,
  "info": {
    "grade": 0.84,
    "component_scores": {
      "esi_accuracy": 1.0,
      "rationale_keywords": 0.87,
      "intervention_recall": 0.75
    },
    "critical_errors": [],
    "passed": true,
    "total_reward": 0.84,
    "task_id": "triage_medium",
    "difficulty": "medium"
  }
}
```

---

## 🧪 Running the Inference Benchmark

The `inference.py` module provides a complete benchmark runner that connects any OpenAI-compatible model endpoint to the environment.

```bash
# Run all 9 tasks with the default model
export HF_TOKEN=your_hf_token
python inference.py

# Run specific tasks
python inference.py --tasks triage_hard med_safety_hard sepsis_hard

# Use a different model
python inference.py --model meta-llama/Llama-3.1-405B-Instruct

# Disable chain-of-thought (faster, usually lower scores)
python inference.py --no-cot

# Save results to JSON + CSV
python inference.py --output results/benchmark_run.json
```

### Chain-of-Thought Prompting

The inference engine uses domain-specific system prompts with **structured CoT**:

```
REASONING: [3-5 sentences of clinical thinking]
ACTION: {"esi_level": 2, "rationale": "...", ...}
```

This forces the model to explicitly reason before acting, which significantly improves scores — especially on hard tasks where missing a single clinical detail causes cascading errors.

### Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token |
| `MODEL_NAME` | `meta-llama/Llama-3.3-70B-Instruct` | Model to evaluate |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible endpoint |
| `USE_COT` | `true` | Enable chain-of-thought |

---

## 🏆 Meta × Scaler Open-Env Hackathon Alignment

ClinicalTriageEnv was designed specifically to satisfy all hackathon evaluation criteria:

### ✅ Criterion 1: OpenEnv API Compliance
- Full `reset()` / `step()` / `state()` interface implemented in `environment.py`
- `openenv.yaml` manifest with complete metadata, baseline scores, and server config
- Pydantic v2-typed action and observation spaces with field-level validation
- Continuous reward signal in `[-1.0, 1.5]` range with shaped components

### ✅ Criterion 2: Domain Meaningfulness & Real-World Impact
- Three distinct medical domains grounded in real clinical protocols
- Scenarios reflect actual patient presentations seen in EDs and ICUs globally
- Graders encode published guidelines (ESI v4, SSC 2021, FDA drug interactions)
- Every wrong answer has a quantifiable real-world consequence model

### ✅ Criterion 3: Task Difficulty Gradient
- 3 difficulty levels per domain (Easy / Medium / Hard) across 9 total tasks
- Baseline LLM scores decline sharply with difficulty: `0.90 → 0.31` in medication safety
- Hard tasks require multi-step clinical reasoning that even 70B models fail

### ✅ Criterion 4: Programmatic Grading Quality
- No LLM-as-judge; fully deterministic, auditable, reproducible graders
- 7–9 component axes per domain with calibrated weights summing to 1.0
- Partial credit architecture rewards incremental clinical improvement
- Critical patient-safety errors enforced with hard score caps

### ✅ Criterion 5: Infrastructure & Deployment
- Dockerized on `python:3.11-slim`; deploys on HuggingFace Spaces with zero configuration
- FastAPI REST + WebSocket server on port 7860
- Interactive HTML UI for human testing and live demos
- Complete inference benchmark script with JSON + CSV output

### ✅ Criterion 6: Code Quality & Documentation
- Fully type-annotated Python 3.11+ codebase with Pydantic v2 models
- Modular separation: `env / models / graders / scenarios / inference / app`
- `openenv.yaml` with baseline scores, episode configuration, and server specification
- Comprehensive docstrings on every class and method

---

## 🔮 Future Roadmap

| Feature | Priority | Description |
|---------|----------|-------------|
| **Multi-turn episodes** | High | Allow agents to order labs, request consults, then revise decisions |
| **Expanded scenario library** | High | 50+ scenarios across 10+ additional clinical domains |
| **Pediatric & OB tracks** | Medium | Child-specific vital norms, obstetric emergencies |
| **Time-pressure simulation** | Medium | Real-time clock; reward decay for slow decisions |
| **Diagnostic reasoning tasks** | Medium | Differential diagnosis generation + workup planning |
| **RLHF training integration** | High | Reward model export for PPO/GRPO fine-tuning pipelines |
| **Multi-agent simulation** | Low | Nurse + physician + pharmacist agent collaboration |
| **EHR integration** | Low | FHIR-compatible patient record format |

---

## 📦 Dependencies

```
fastapi
uvicorn
pydantic>=2.0
openai
python-multipart
```

---

## ⚠️ Clinical Disclaimer

ClinicalTriageEnv is a research and training tool for AI systems. **It is not a substitute for professional medical judgment.** The scenarios, graders, and feedback are designed to train AI agents — they should never be used to make actual patient care decisions. All medical content is based on published clinical guidelines but has not been validated by licensed medical professionals for use in clinical practice.

---

## 📜 License

MIT License — open for research, extension, and benchmarking.

---

<div align="center">

**Built with ❤️ for the Meta × Scaler Open-Env Hackathon**

*Advancing AI safety in healthcare, one graded episode at a time.*

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Try%20It%20Live-ClinicalTriageEnv-ff9f00?style=for-the-badge)](https://huggingface.co/spaces/samrudh-nux/ClinicalTriageEnv)

</div>


