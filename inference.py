"""
inference.py — ClinicalTriageEnv Baseline Inference Script
===========================================================
Runs an LLM agent against all 9 tasks (3 difficulty levels × 3 task types)
and reports reproducible baseline scores.
Environment variables required:
  API_BASE_URL - LLM API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME  - Model identifier for inference
  HF_TOKEN - Hugging Face / API key
Usage:
  python inference.py
  python inference.py --tasks triage_easy triage_medium sepsis_hard
  python inference.py --output results.json
  (shin )
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional
from datetime import datetime

  from openai import OpenAI

       # Add parent directory to path for local imports
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import ClinicalTriageEnv, TASK_REGISTRY
               from models import (
    TriageAction, MedicationSafetyAction, SepsisManagementAction
)

# 
# Configuration
# 

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
               API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
      MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

TEMPERATURE = 0.1  # Low temp for reproducible results
MAX_TOKENS = 1200
MAX_RETRIES = 3

# All 9 tasks ordered by difficulty
ALL_TASKS = [
    "triage_easy",
    "triage_medium",
    "triage_hard",
    "med_safety_easy",
    "med_safety_medium",
    "med_safety_hard",
    "sepsis_easy",
    "sepsis_medium",
    "sepsis_hard",
]


#
# LLM Client
# 

def get_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "dummy-key",
    )


def call_llm(client: OpenAI, prompt: str, system: str = "") -> str:
    """Call LLM with retry logic."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ⚠ LLM call failed after {MAX_RETRIES} attempts: {e}")
                return ""
    return ""


# 
# Task-specific prompts and action parsers
# 

SYSTEM_PROMPT = """You are a highly experienced clinical AI assistant with expertise in emergency medicine, pharmacology, and critical care.
You will be given a clinical scenario and must respond with a JSON object matching the required action format.
CRITICAL: Respond ONLY with a valid JSON object. No markdown, no explanation, just JSON.
Your responses will be evaluated for medical accuracy and patient safety."""


def build_triage_prompt(obs) -> str:
    p = obs.patient
    v = p.vitals
    return f"""## EMERGENCY DEPARTMENT TRIAGE TASK
               **Task**: {obs.task_description}
                 **Patient Record**:
- Patient ID: {p.patient_id}
- Age/Sex: {p.age}y {p.sex}
- Chief Complaint: {p.chief_complaint}
**Vital Signs**:
- Heart Rate: {v.heart_rate} bpm
- Blood Pressure: {v.systolic_bp}/{v.diastolic_bp} mmHg
            - Temperature: {v.temperature}°C
- SpO2: {v.spo2}%
- Respiratory Rate: {v.respiratory_rate}/min
- GCS: {v.glasgow_coma_scale}/15
**Symptoms**: {', '.join(p.symptoms)}
**Medical History**: {', '.join(p.medical_history)}
**Current Medications**: {', '.join([f"{m.name} {m.dose_mg}mg {m.frequency}" for m in p.current_medications]) or 'None'}
**Allergies**: {', '.join(p.allergies) or 'NKDA'}
**Lab Results**: {json.dumps(p.lab_results) if p.lab_results else 'Pending'}
**ESI Scale Reference**:
- ESI-1: Requires immediate life-saving intervention
- ESI-2: High risk situation, severe pain, or vital sign abnormality
- ESI-3: Stable, requires 2+ resources (labs, imaging, IV)
- ESI-4: Stable, requires 1 resource
- ESI-5: No resources needed
Respond with JSON:
{{
  "esi_level": <1-5>,
  "rationale": "<clinical reasoning, minimum 30 words>",
  "recommended_immediate_interventions": ["<intervention1>", "<intervention2>"]
}}"""


def build_med_safety_prompt(obs) -> str:
    p = obs.patient
    meds_str = "\n".join([
        f"  - {m.name}: {m.dose_mg}mg {m.frequency} ({m.route})"
        for m in p.current_medications
    ])
    labs_str = json.dumps(p.lab_results, indent=2) if p.lab_results else "No labs"

    drug_info_str = ""
    if obs.available_drug_info:
        drug_info_str = f"\n**Drug Reference Information**:\n{json.dumps(obs.available_drug_info, indent=2)}"

    return f"""## MEDICATION SAFETY REVIEW TASK
**Task**: {obs.task_description}
**Patient**:
- Age/Sex: {p.age}y {p.sex}
- Medical History: {', '.join(p.medical_history)}
- Allergies: {', '.join(p.allergies) or 'NKDA'}
**Current Medications**:
{meds_str}
**Laboratory Results**:
{labs_str}
**Chief Complaint/Context**: {p.chief_complaint}
{drug_info_str}
**Severity Scale**: safe < minor < moderate < major < critical
Respond with JSON:
{{
  "flagged_interactions": ["<drug1+drug2: mechanism/risk>"],
  "flagged_contraindications": ["<drug_condition: reason>"],
  "flagged_dosing_errors": ["<drug_dose_error: correction>"],
  "recommended_changes": ["<specific change>"],
  "severity_assessment": "<safe|minor|moderate|major|critical>",
  "clinical_rationale": "<detailed clinical explanation, minimum 50 words>"
}}"""


def build_sepsis_prompt(obs) -> str:
    p = obs.patient
    v = p.vitals
    labs = json.dumps(p.lab_results, indent=2) if p.lab_results else "Pending"
    meds = ", ".join([f"{m.name} {m.dose_mg}mg" for m in p.current_medications]) or "None"

    return f"""## SEPSIS RECOGNITION & MANAGEMENT TASK
**Task**: {obs.task_description}
**Patient**:
- Age/Sex: {p.age}y {p.sex}
- Chief Complaint: {p.chief_complaint}
- Allergies: {', '.join(p.allergies) or 'NKDA'}
- Medical History: {', '.join(p.medical_history)}
- Current Medications: {meds}
**Vital Signs**:
- HR: {v.heart_rate}, BP: {v.systolic_bp}/{v.diastolic_bp}, Temp: {v.temperature}°C
- SpO2: {v.spo2}%, RR: {v.respiratory_rate}, GCS: {v.glasgow_coma_scale}/15
- MAP: {int((v.systolic_bp + 2*v.diastolic_bp)/3)} mmHg
**qSOFA Score (pre-calculated)**: {obs.qsofa_score}/3
- (RR≥22: 1pt, altered mentation GCS<15: 1pt, SBP≤100: 1pt)
**Laboratory Results**:
{labs}
**Time elapsed**: {obs.time_elapsed_minutes} minutes since arrival
**Sepsis Definitions (Sepsis-3)**:
- Sepsis: Life-threatening organ dysfunction from infection (SOFA≥2)
- Septic Shock: Sepsis + vasopressors needed + lactate>2 despite fluids
**Hour-1 SSC Bundle (complete ALL applicable)**:
1. Blood cultures BEFORE antibiotics
2. Broad-spectrum antibiotics within 1 hour
3. Lactate measurement (repeat if >2)
4. 30mL/kg crystalloid if hypotension or lactate≥4
5. Vasopressors if MAP<65 despite fluids
Respond with JSON:
{{
  "sepsis_diagnosis": "<sepsis|septic_shock|SIRS_only|no_sepsis>",
  "blood_cultures_ordered": <true|false>,
  "antibiotics_ordered": <true|false>,
  "antibiotic_choice": "<drug_name or null>",
  "lactate_ordered": <true|false>,
  "iv_fluid_bolus_ml": <int>,
  "vasopressor_ordered": <true|false>,
  "vasopressor_choice": "<drug_name or null>",
  "source_control_identified": "<source or null>",
  "clinical_rationale": "<detailed reasoning, minimum 50 words>",
  "time_to_antibiotics_minutes": <int or null>
}}"""


def parse_llm_response(response: str, task_type: str):
    """Parse LLM JSON response into the appropriate action type."""
    if not response:
        return _get_fallback_action(task_type)

    # Clean up response (strip markdown code blocks if present)
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                return _get_fallback_action(task_type)
        else:
            return _get_fallback_action(task_type)

    try:
        if task_type == "triage":
            return TriageAction(**data)
        elif task_type == "medication_safety":
            return MedicationSafetyAction(**data)
        elif task_type == "sepsis":
            return SepsisManagementAction(**data)
    except Exception as e:
        print(f"  ⚠ Action parsing error: {e}")
        return _get_fallback_action(task_type)


def _get_fallback_action(task_type: str):
    """Return a neutral fallback action when LLM fails."""
    if task_type == "triage":
        return TriageAction(esi_level=3, rationale="Unable to parse response - defaulting to ESI-3")
    elif task_type == "medication_safety":
        return MedicationSafetyAction(
            severity_assessment="moderate",
            clinical_rationale="Unable to parse response - review required"
        )
    elif task_type == "sepsis":
        return SepsisManagementAction(
            sepsis_diagnosis="sepsis",
            clinical_rationale="Unable to parse response - standard sepsis protocol",
            blood_cultures_ordered=True,
            antibiotics_ordered=True,
            lactate_ordered=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str, verbose: bool = True) -> Dict[str, Any]:
    """Run a single task and return results."""
    env = ClinicalTriageEnv(task_id=task_id)
    task_meta = TASK_REGISTRY[task_id]
    task_type = task_meta["type"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_meta['name']}")
        print(f"Type: {task_type} | Difficulty: {task_meta['difficulty']}")
        print(f"{'='*60}")

    # Reset environment
    obs = env.reset()

    if verbose:
        print(f"Patient: {obs.patient.chief_complaint}")
        print(f"Task: {obs.task_description[:120]}...")

    # Build prompt based on task type
    if task_type == "triage":
        prompt = build_triage_prompt(obs)
    elif task_type == "medication_safety":
        prompt = build_med_safety_prompt(obs)
    elif task_type == "sepsis":
        prompt = build_sepsis_prompt(obs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Call LLM
    if verbose:
        print(f"\nCalling {MODEL_NAME}...")
    
    start = time.time()
    llm_response = call_llm(client, prompt, SYSTEM_PROMPT)
    elapsed = time.time() - start

    if verbose and llm_response:
        print(f"LLM Response ({elapsed:.1f}s):\n{llm_response[:500]}{'...' if len(llm_response) > 500 else ''}")

    # Parse action
    action = parse_llm_response(llm_response, task_type)

    if verbose:
        print(f"\nParsed Action: {action.model_dump()}")

    # Step environment
    obs_out, reward, done, info = env.step(action)

    if verbose:
        print(f"\n{'-'*50}")
        print(f"Score: {info['grade']:.3f} | Reward: {reward:.3f}")
        print(f"Passed: {'✅' if info['passed'] else '❌'}")
        if info.get("critical_errors"):
            print(f"Critical Errors: {info['critical_errors']}")
        print(f"\nFeedback:\n{obs_out.feedback}")

    return {
        "task_id": task_id,
        "task_name": task_meta["name"],
        "task_type": task_type,
        "difficulty": task_meta["difficulty"],
        "score": info["grade"],
        "reward": reward,
        "passed": info["passed"],
        "component_scores": info["component_scores"],
        "critical_errors": info["critical_errors"],
        "llm_response_length": len(llm_response),
        "inference_time_seconds": round(elapsed, 2),
    }


def run_all_tasks(task_ids: List[str], verbose: bool = True) -> Dict[str, Any]:
    """Run all tasks and produce a summary report."""
    client = get_client()
    results = []
    total_start = time.time()

    print(f"\n{'#'*60}")
    print(f"# ClinicalTriageEnv Baseline Inference")
    print(f"# Model: {MODEL_NAME}")
    print(f"# API: {API_BASE_URL}")
    print(f"# Tasks: {len(task_ids)}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    for task_id in task_ids:
        try:
            result = run_task(client, task_id, verbose=verbose)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Task {task_id} failed: {e}")
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "reward": 0.0,
                "passed": False,
                "error": str(e),
            })
        # Small delay between tasks to avoid rate limiting
        time.sleep(1)

    total_time = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────────
    scores_by_type = {}
    scores_by_difficulty = {}

    for r in results:
        t = r.get("task_type", "unknown")
        d = r.get("difficulty", "unknown")
        scores_by_type.setdefault(t, []).append(r["score"])
        scores_by_difficulty.setdefault(d, []).append(r["score"])

    overall_scores = [r["score"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    summary = {
        "run_info": {
            "model": MODEL_NAME,
            "api_base": API_BASE_URL,
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 1),
            "num_tasks": len(task_ids),
        },
        "overall": {
            "mean_score": round(overall_avg, 4),
            "pass_rate": round(sum(1 for r in results if r.get("passed")) / len(results), 3),
            "scores": {r["task_id"]: r["score"] for r in results},
        },
        "by_task_type": {
            t: round(sum(s)/len(s), 4) for t, s in scores_by_type.items()
        },
        "by_difficulty": {
            d: round(sum(s)/len(s), 4) for d, s in scores_by_difficulty.items()
        },
        "detailed_results": results,
    }

    # Print summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<35} {'Score':>7} {'Reward':>8} {'Pass':>6}")
    print(f"{'-'*60}")
    for r in results:
        name = r.get("task_name", r["task_id"])[:33]
        score = r.get("score", 0.0)
        reward = r.get("reward", 0.0)
        passed = "✅" if r.get("passed") else "❌"
        print(f"{name:<35} {score:>7.3f} {reward:>8.4f} {passed:>6}")

    print(f"{'-'*60}")
    print(f"{'OVERALL MEAN':<35} {overall_avg:>7.3f}")
    print(f"\nBy Task Type:")
    for t, avg in summary["by_task_type"].items():
        print(f"  {t:<30} {avg:.4f}")
    print(f"\nBy Difficulty:")
    for d, avg in summary["by_difficulty"].items():
        print(f"  {d:<30} {avg:.4f}")
    print(f"\nTotal inference time: {total_time:.1f}s")
    print(f"{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ClinicalTriageEnv Baseline Inference Script"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ALL_TASKS,
        choices=list(TASK_REGISTRY.keys()),
        help="Tasks to run (default: all 9 tasks)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file for results (default: results.json)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    # Validate credentials
    if not API_KEY:
        print("⚠️  Warning: No API key found. Set HF_TOKEN or API_KEY environment variable.")
        print("   Continuing with dummy key (will fail on real API calls)...")

    # Run inference
    results = run_all_tasks(args.tasks, verbose=not args.quiet)

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {output_path}")

    # Exit code: 0 if pass rate > 0, 1 if all failed
    pass_rate = results["overall"]["pass_rate"]
    sys.exit(0 if pass_rate > 0 else 1)


if __name__ == "__main__":
    main()
