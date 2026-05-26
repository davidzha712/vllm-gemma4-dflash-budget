#!/usr/bin/env python3
"""Verify that thinking_token_budget is actually enforced by this image.

Hits /v1/chat/completions with a reasoning-heavy prompt at several budgets and
prints how reasoning_content length scales. Monotonic increase from small to
medium budgets means the budget knob is real; flat lengths across budgets mean
one of the patch layers didn't apply.

Usage:
    python3 verify-budget.py <endpoint> [--model NAME] [--auth TOKEN]

Example:
    python3 verify-budget.py http://localhost:8000 --model gemma4
"""
import argparse
import json
import sys
import urllib.request

PROMPT = (
    "Prove that the sum of two odd numbers is always even. "
    "Show every step of the algebraic reasoning."
)

BUDGETS = [10, 50, 200, 1000, 10000]


def probe(endpoint: str, model: str, auth: str | None, budget: int) -> dict:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "thinking_token_budget": budget,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        j = json.loads(r.read())
    msg = (j.get("choices") or [{}])[0].get("message", {})
    usage = j.get("usage") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    return {
        "budget": budget,
        "reason_chars": len(reasoning),
        "reason_words": len(reasoning.split()),
        "content_chars": len(content),
        "total_tokens": usage.get("completion_tokens"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("endpoint", help="e.g. http://localhost:8000")
    p.add_argument("--model", default="gemma4-dflash-budget")
    p.add_argument("--auth", help="Bearer token, if your endpoint requires one")
    args = p.parse_args()

    print(f"{'budget':>8}  {'reason_chars':>12}  {'reason_words':>11}  "
          f"{'content_chars':>13}  {'tot_tok':>7}")
    print("-" * 64)

    results = []
    for b in BUDGETS:
        try:
            r = probe(args.endpoint, args.model, args.auth, b)
            print(f"{r['budget']:>8}  {r['reason_chars']:>12}  "
                  f"{r['reason_words']:>11}  {r['content_chars']:>13}  "
                  f"{str(r['total_tokens']):>7}")
            results.append(r)
        except Exception as e:
            print(f"{b:>8}  ERROR: {type(e).__name__}: {str(e)[:100]}")
            return 2

    # Sanity check — reasoning length at budget=10 should be << budget=1000.
    small = next((r["reason_chars"] for r in results if r["budget"] == 10), None)
    big = next((r["reason_chars"] for r in results if r["budget"] == 1000), None)
    if small is None or big is None:
        return 3
    if big < small * 5:
        print("\nFAIL: reasoning length did not scale with budget.")
        print("      budget=10 produced", small, "chars; budget=1000 produced", big)
        print("      thinking_token_budget is NOT being enforced.")
        return 1

    print("\nOK — thinking_token_budget is being enforced "
          f"(budget=10 → {small} chars; budget=1000 → {big} chars).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
