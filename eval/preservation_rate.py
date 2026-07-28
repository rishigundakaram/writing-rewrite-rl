"""Content-preservation rate per system on the 128 clean-val h2h prompts.

Scores every system's rewrite (base, our three RL models, and the gpt-5.6
reference) against its source with the training preservation judge
(writing_rewrite._gate's prompt/schema). Reports, per system:
  - preservation rate: fraction of rewrites the judge finds zero issues in
  - clean-or-minor rate: fraction with no meaning_change (omissions only)
  - mean gate: mean 0.5**weighted, the multiplicative factor training used
  - mean issues per rewrite

Per-(system, prompt) verdicts cache to outputs/preservation.json. Strong
judge by default (gpt-5); override with PRESERVE_MODEL.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys

from openai import AsyncOpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "environments" / "writing_rewrite"))
import writing_rewrite as wr  # noqa: E402

ITEMS = ROOT / "outputs" / "h2h2" / "items"
CACHE = ROOT / "outputs" / "preservation.json"
JUDGE = os.environ.get("PRESERVE_MODEL", "gpt-5")
SYSTEMS = ["gpt-5.6", "ranked", "ranked_attribute", "absolute_attribute", "base"]


async def judge_one(client: AsyncOpenAI, source: str, rewrite: str) -> dict:
    """One preservation verdict -> {issues, meaning_changes, weighted, gate}."""
    if not rewrite.strip():
        return {"issues": 99, "meaning_changes": 99, "weighted": 99, "gate": 0.0}
    resp = await client.chat.completions.parse(
        model=JUDGE,
        messages=[{"role": "user",
                   "content": wr.PRESERVE_PROMPT.format(source=source, rewrite=rewrite)}],
        response_format=wr.PreserveVerdict,
    )
    verdict = resp.choices[0].message.parsed
    if verdict is None:
        return {"issues": 99, "meaning_changes": 99, "weighted": 99, "gate": 0.0}
    mc = sum(1 for i in verdict.issues if i.type == "meaning_change")
    weighted = sum(2 if i.type == "meaning_change" else 1 for i in verdict.issues)
    return {"issues": len(verdict.issues), "meaning_changes": mc,
            "weighted": weighted, "gate": 0.5 ** weighted}


async def main() -> None:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(int(os.environ.get("PRESERVE_CONCURRENCY", "40")))
    items = sorted(ITEMS.glob("*.json"))

    async def score(system: str, item_path: pathlib.Path) -> None:
        key = f"{JUDGE}::{system}::{item_path.stem}"
        if key in cache:
            return
        item = json.loads(item_path.read_text())
        by_key = {v["key"]: v["text"] for v in item["labels"].values()}
        async with sem:
            try:
                cache[key] = await judge_one(client, item["source"], by_key[system])
                CACHE.write_text(json.dumps(cache, indent=0))  # incremental: survives interruption
            except Exception as e:  # noqa: BLE001 - record gap, rerun fills it
                print(f"  {key} failed: {str(e)[:80]}")

    await asyncio.gather(*(score(s, p) for s in SYSTEMS for p in items))
    CACHE.write_text(json.dumps(cache, indent=0))

    print(f"\n=== content preservation ({JUDGE} judge, n={len(items)}) ===")
    print(f"{'system':20s} {'preserve%':>9} {'no-mean-chg%':>12} {'mean gate':>10} {'issues/rw':>10}")
    for system in SYSTEMS:
        vs = [cache[f"{JUDGE}::{system}::{p.stem}"] for p in items
              if f"{JUDGE}::{system}::{p.stem}" in cache]
        n = len(vs)
        if not n:
            continue
        clean = sum(v["issues"] == 0 for v in vs) / n
        no_mc = sum(v["meaning_changes"] == 0 for v in vs) / n
        gate = sum(v["gate"] for v in vs) / n
        iss = sum(v["issues"] for v in vs) / n
        print(f"{system:20s} {clean*100:8.1f}% {no_mc*100:11.1f}% {gate:10.3f} {iss:10.2f}")


if __name__ == "__main__":
    asyncio.run(main())
