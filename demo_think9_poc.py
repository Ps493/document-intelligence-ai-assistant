"""
demo_think9_poc.py
-------------------
Runs the Think9 PoC demo end-to-end against a locally running api.py
(python api.py, in a separate terminal, on http://localhost:5000).

This mirrors the 4-step demo script in the architecture document:
  1. Ask the Memory Agent a question answerable from ingested docs.
  2. Ask a question OUTSIDE the corpus -> should decline, not hallucinate.
  3. Trigger the Sourcing Agent on the same shared memory -> cross-brand insight.
  4. (Talk through the code for step 4 - confidence gate / human-in-loop -
     this script prints the confidence field so you can point at it live.)

Usage:
    python api.py                  # terminal 1
    python demo_think9_poc.py      # terminal 2
"""

import requests

BASE = "http://localhost:5000"

CORPUS = [
    ("sample_data/think9_corpus/brandorbit_vendor_notes.txt", "BrandOrbit", "vendor_notes"),
    ("sample_data/think9_corpus/brandorbit_playbook.txt", "BrandOrbit", "playbook"),
    ("sample_data/think9_corpus/luneskin_vendor_notes.txt", "LuneSkin", "vendor_notes"),
    ("sample_data/think9_corpus/luneskin_meeting_notes.txt", "LuneSkin", "meeting_notes"),
]


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def ingest_corpus():
    line("STEP 0 — Ingesting Think9 demo corpus into shared memory")
    for path, brand, function in CORPUS:
        with open(path, "rb") as f:
            resp = requests.post(
                f"{BASE}/upload",
                files={"file": f},
                data={"brand": brand, "function": function},
            )
        print(f"  Uploaded {path}  ->  {resp.json()}")


def step1_confident_answer():
    line("STEP 1 — Institutional Memory Agent: confident, cited answer")
    q = "What is the agreed MOQ and lead time with Sunrise Packaging Co for BrandOrbit?"
    resp = requests.post(f"{BASE}/ask", json={"question": q, "brand": "BrandOrbit"})
    data = resp.json()
    print(f"Q: {q}")
    print(f"Confidence: {data.get('confidence')}")
    print(f"A: {data.get('answer')}")


def step2_decline_out_of_corpus():
    line("STEP 2 — Confidence gate: question OUTSIDE the ingested corpus")
    q = "What is BrandOrbit's projected revenue for fiscal year 2028?"
    resp = requests.post(f"{BASE}/ask", json={"question": q})
    data = resp.json()
    print(f"Q: {q}")
    print(f"Confidence: {data.get('confidence')}  |  needs_human_review: {data.get('needs_human_review')}")
    print(f"A: {data.get('answer')}")


def step3_sourcing_agent_cross_brand():
    line("STEP 3 — Sourcing Agent (pluggable, same core): cross-brand insight")
    q = "Are there any vendors shared across brands where we could bundle volume for a better rate?"
    resp = requests.post(f"{BASE}/sourcing_ask", json={"query": q})
    data = resp.json()
    print(f"Q: {q}")
    print(f"Confidence: {data.get('confidence')}")
    print(f"A: {data.get('answer')}")


def step4_talking_point():
    line("STEP 4 — Talking point (explain in your own words during the demo)")
    print("The confidence/needs_human_review fields you just saw in Steps 1-3 are the")
    print("human-in-the-loop gate from the architecture doc: low-confidence output is")
    print("routed for review instead of being sent to a person as a guessed answer.")


if __name__ == "__main__":
    ingest_corpus()
    step1_confident_answer()
    step2_decline_out_of_corpus()
    step3_sourcing_agent_cross_brand()
    step4_talking_point()
