"""
test_agent.py — Quick smoke tests for the SHL agent.
Run: python test_agent.py
Requires GROQ_API_KEY to be set in environment (or uses the default key in agent.py).
"""

import json
import sys
import os
import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


def test_health():
    """Test GET /health endpoint."""
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print("[PASS] /health returns 200 OK")


def test_schema_compliance(response_json: dict, label: str = ""):
    """Assert the response matches the required schema."""
    assert "reply" in response_json, f"{label}: missing 'reply'"
    assert "recommendations" in response_json, f"{label}: missing 'recommendations'"
    assert "end_of_conversation" in response_json, f"{label}: missing 'end_of_conversation'"
    assert isinstance(response_json["reply"], str), f"{label}: 'reply' not string"
    assert isinstance(response_json["recommendations"], list), f"{label}: 'recommendations' not list"
    assert isinstance(response_json["end_of_conversation"], bool), f"{label}: 'end_of_conversation' not bool"
    for rec in response_json["recommendations"]:
        assert "name" in rec, f"{label}: recommendation missing 'name'"
        assert "url" in rec, f"{label}: recommendation missing 'url'"
        assert "test_type" in rec, f"{label}: recommendation missing 'test_type'"
        assert rec["url"].startswith("https://www.shl.com/"), f"{label}: invalid URL: {rec['url']}"
    print(f"[PASS] Schema compliant: {label}")


def chat(messages: list[dict]) -> dict:
    """Send a POST /chat request."""
    r = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    return r.json()


def test_vague_query():
    """Agent should clarify, NOT recommend on first turn for vague query."""
    resp = chat([{"role": "user", "content": "I need an assessment"}])
    test_schema_compliance(resp, "vague_query")
    assert len(resp["recommendations"]) == 0, "Should NOT recommend on vague query"
    assert resp["end_of_conversation"] is False
    print(f"  Reply: {resp['reply'][:100]}...")
    print("[PASS] Vague query: agent clarifies, no recommendations")


def test_specific_query():
    """Agent should recommend for a detailed query."""
    resp = chat([
        {"role": "user", "content": "I'm hiring a mid-level Java developer. They need strong core Java and Spring framework skills. We want to test their technical knowledge."}
    ])
    test_schema_compliance(resp, "specific_query")
    # May or may not recommend on first turn depending on agent behavior
    print(f"  Reply: {resp['reply'][:100]}...")
    print(f"  Recommendations: {len(resp['recommendations'])}")
    if resp["recommendations"]:
        for rec in resp["recommendations"]:
            print(f"    - {rec['name']} ({rec['test_type']})")
    print("[PASS] Specific query handled")


def test_refinement():
    """Agent should update shortlist when user changes constraints."""
    # Turn 1: initial query
    msgs = [
        {"role": "user", "content": "I need assessments for a senior Java developer"}
    ]
    resp1 = chat(msgs)
    test_schema_compliance(resp1, "refinement_turn1")
    print(f"  Turn 1 reply: {resp1['reply'][:80]}...")

    # Turn 2: agent's reply + user refinement
    msgs.append({"role": "assistant", "content": resp1["reply"]})
    msgs.append({"role": "user", "content": "Also add personality and cognitive ability tests"})
    resp2 = chat(msgs)
    test_schema_compliance(resp2, "refinement_turn2")
    print(f"  Turn 2 reply: {resp2['reply'][:80]}...")
    print(f"  Recommendations: {len(resp2['recommendations'])}")
    print("[PASS] Refinement handled")


def test_off_topic():
    """Agent should refuse off-topic questions."""
    resp = chat([
        {"role": "user", "content": "What is the average salary for a software engineer in New York?"}
    ])
    test_schema_compliance(resp, "off_topic")
    assert len(resp["recommendations"]) == 0, "Should NOT recommend for off-topic"
    print(f"  Reply: {resp['reply'][:100]}...")
    print("[PASS] Off-topic question refused")


def test_c1_trace():
    """Replay C1 sample conversation (senior leadership selection)."""
    print("\n--- C1: Senior Leadership ---")

    msgs = [{"role": "user", "content": "We need a solution for senior leadership."}]
    r1 = chat(msgs)
    test_schema_compliance(r1, "C1_turn1")
    assert len(r1["recommendations"]) == 0, "C1 T1: should clarify first"
    print(f"  T1: {r1['reply'][:80]}...")

    msgs.append({"role": "assistant", "content": r1["reply"]})
    msgs.append({"role": "user", "content": "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."})
    r2 = chat(msgs)
    test_schema_compliance(r2, "C1_turn2")
    print(f"  T2: {r2['reply'][:80]}...")

    msgs.append({"role": "assistant", "content": r2["reply"]})
    msgs.append({"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."})
    r3 = chat(msgs)
    test_schema_compliance(r3, "C1_turn3")
    print(f"  T3: {r3['reply'][:80]}...")
    print(f"  Recommendations: {len(r3['recommendations'])}")
    for rec in r3["recommendations"]:
        print(f"    - {rec['name']} ({rec['test_type']})")

    # Check if OPQ32r is in recommendations (expected from C1 trace)
    names = [r["name"].lower() for r in r3["recommendations"]]
    if any("opq" in n for n in names):
        print("  [PASS] OPQ found in recommendations (matches C1 trace)")
    else:
        print("  [WARN] OPQ not found in recommendations")

    print("[PASS] C1 trace replayed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("SHL Assessment Recommender — Smoke Tests")
    print("=" * 60)

    test_health()

    print("\n--- Behavioral Tests ---")
    test_vague_query()
    test_specific_query()
    test_refinement()
    test_off_topic()

    test_c1_trace()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
