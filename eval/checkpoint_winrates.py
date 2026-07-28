"""Dense winrate curves: every checkpoint adapter x all 128 cached prompts.

For each checkpoint adapter of the three clean-split runs (steps 40..400)
plus the untrained base model: deploy if needed, generate rewrites for all
128 winrate prompts, judge each pairwise vs the cached gpt-5.6 reference
(same judge + flip logic as the in-training eval), and record win rates.

Per-(model, prompt) results cache to outputs/winrate_dense.json so reruns
only fill gaps. Needs OPENAI_API_KEY (judge) + prime CLI auth (deploy +
policy generations).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import subprocess
import sys

from openai import AsyncOpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "environments" / "writing_rewrite"))
import writing_rewrite as wr  # noqa: E402

PRIME = str(pathlib.Path.home() / ".local" / "bin" / "prime")
REFS = ROOT / "environments" / "writing_rewrite" / "writing_rewrite" / "winrate_refs.jsonl"
CACHE = ROOT / "outputs" / "winrate_dense.json"
PRIME_BASE = "https://api.pinference.ai/api/v1"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
RUN_NAMES = ("ranked", "absolute_attribute", "ranked_attribute")


def _prime_key() -> str:
    return json.load(open(pathlib.Path.home() / ".prime" / "config.json"))["api_key"]


def list_adapters() -> list[dict]:
    """All checkpoint adapters for the three named runs, via the CLI."""
    rows = []
    for page in (1, 2, 3, 4):
        out = subprocess.run(
            [PRIME, "deployments", "list", "--plain", "--page", str(page)],
            capture_output=True, text=True, env={"COLUMNS": "300", **__import__("os").environ},
        ).stdout
        for line in out.splitlines():
            m = re.match(r"(\w{20,26})\s+(\S+)\s+meta-llama\S*\s+(\d+|-)\s+\[\w+\](\w+)", line.strip())
            if m and m.group(2) in RUN_NAMES and m.group(3) != "-":
                rows.append({
                    "adapter_id": m.group(1), "run": m.group(2),
                    "step": int(m.group(3)), "deployed": m.group(4) == "DEPLOYED",
                })
        if "page" not in out.lower() or f"page {page} of" not in out.lower():
            if page > 1 and not out.strip():
                break
    # de-dup (step 399/400 both exist near run end; keep multiples of 40 only)
    return [r for r in rows if r["step"] % 40 == 0]


def ensure_deployed(adapters: list[dict]) -> None:
    for a in adapters:
        if not a["deployed"]:
            subprocess.run([PRIME, "deployments", "create", a["adapter_id"], "--plain", "-y"],
                           capture_output=True, text=True)
    # poll until all deployed
    import time

    for _ in range(60):
        current = {x["adapter_id"]: x["deployed"] for x in list_adapters()}
        if all(current.get(a["adapter_id"], False) for a in adapters):
            return
        time.sleep(30)
    raise RuntimeError("adapters did not all reach DEPLOYED within 30 minutes")


async def eval_model(name: str, model_id: str, refs: list[dict], cache: dict) -> float:
    policy = AsyncOpenAI(api_key=_prime_key(), base_url=PRIME_BASE)
    judge = AsyncOpenAI()
    sem = asyncio.Semaphore(8)

    async def one(row: dict) -> float:
        key = f"{name}::{row['id']}"
        if key in cache:
            return cache[key]
        async with sem:
            try:
                r = await policy.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": wr.INSTRUCTION + row["source"]}],
                    max_tokens=700,
                )
                rewrite = (r.choices[0].message.content or "").strip()
                score = await wr._winrate_score(judge, row["id"], row["source"], row["ref"], rewrite)
            except Exception as e:  # noqa: BLE001 - record and continue; rerun fills gaps
                print(f"  {key} failed: {str(e)[:80]}")
                return -1.0
        cache[key] = score
        return score

    scores = [s for s in await asyncio.gather(*(one(r) for r in refs)) if s >= 0]
    CACHE.write_text(json.dumps(cache, indent=0))
    return sum(scores) / len(scores) if scores else float("nan")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="all", help="comma list of run names or 'all'")
    args = parser.parse_args()
    wanted = RUN_NAMES if args.runs == "all" else tuple(args.runs.split(","))

    refs = [json.loads(l) for l in REFS.read_text().splitlines() if l]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    adapters = [a for a in list_adapters() if a["run"] in wanted]
    adapters.sort(key=lambda a: (a["run"], a["step"]))
    print(f"{len(adapters)} checkpoint adapters to evaluate; deploying missing ones...")
    ensure_deployed(adapters)

    results: dict[str, dict[int, float]] = {}
    base_rate = await eval_model("base", BASE_MODEL, refs, cache)
    print(f"base            step 0: {base_rate*100:.1f}%")
    for run in wanted:
        results[run] = {0: base_rate}
    for a in adapters:
        model_id = f"{BASE_MODEL}:{a['adapter_id']}"
        rate = await eval_model(f"{a['run']}@{a['step']}", model_id, refs, cache)
        results[a["run"]][a["step"]] = rate
        print(f"{a['run']:20s} step {a['step']}: {rate*100:.1f}%")

    out = ROOT / "outputs" / "winrate_dense_curves.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
