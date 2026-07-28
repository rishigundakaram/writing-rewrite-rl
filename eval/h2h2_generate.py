"""Head-to-head round 2: 5 systems x 128 clean-val prompts, blind-labeled.

Candidates per prompt: gpt-5.6 (cached ref), base, and the three step-400
finals. Writes outputs/h2h2/items/NNN.json (with system keys, the answer key)
and outputs/h2h2/blind/NNN.json (labels A-E only, judge-visible). Rewrites
regenerate via Prime inference; per-(system,prompt) rewrite cache in
outputs/h2h2/rewrites.json makes reruns cheap.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import random
import sys

from openai import AsyncOpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "environments" / "writing_rewrite"))
import writing_rewrite as wr  # noqa: E402

REFS = ROOT / "environments" / "writing_rewrite" / "writing_rewrite" / "winrate_refs.jsonl"
OUT = ROOT / "outputs" / "h2h2"
PRIME_BASE = "https://api.pinference.ai/api/v1"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
SEED = 20260725

SYSTEMS = {
    "base": BASE_MODEL,
    "ranked": f"{BASE_MODEL}:gabuqxv3n252bz2u3bexj2ux",
    "ranked_attribute": f"{BASE_MODEL}:q793jwjf84dodhpv6ylbhev2",
    "absolute_attribute": None,  # resolved from deployments at runtime
}


def _prime_key() -> str:
    return json.load(open(pathlib.Path.home() / ".prime" / "config.json"))["api_key"]


def _resolve_absolute() -> str:
    import re
    import subprocess
    import os

    out = subprocess.run(
        [str(pathlib.Path.home() / ".local/bin/prime"), "deployments", "list", "--plain"],
        capture_output=True, text=True, env={"COLUMNS": "300", **os.environ},
    ).stdout
    for line in out.splitlines():
        if "absolute_attribute" in line and "NOT_DEPLOYED" not in line and "DEPLOYED" in line:
            return f"{BASE_MODEL}:{line.split()[0]}"
    raise RuntimeError("no deployed absolute_attribute adapter found")


async def main() -> None:
    SYSTEMS["absolute_attribute"] = _resolve_absolute()
    refs = [json.loads(l) for l in REFS.read_text().splitlines() if l]
    (OUT / "items").mkdir(parents=True, exist_ok=True)
    (OUT / "blind").mkdir(exist_ok=True)
    (OUT / "rankings").mkdir(exist_ok=True)
    cache_path = OUT / "rewrites.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client = AsyncOpenAI(api_key=_prime_key(), base_url=PRIME_BASE)
    sem = asyncio.Semaphore(40)

    async def rewrite(system: str, row: dict) -> str:
        key = f"{system}::{row['id']}"
        if key in cache:
            return cache[key]
        async with sem:
            for _ in range(3):
                r = await client.chat.completions.create(
                    model=SYSTEMS[system],
                    messages=[{"role": "user", "content": wr.INSTRUCTION + row["source"]}],
                    max_tokens=700,
                )
                text = (r.choices[0].message.content or "").strip()
                if text:
                    cache[key] = text
                    cache_path.write_text(json.dumps(cache, indent=0))
                    return text
        raise RuntimeError(f"empty rewrite {key}")

    async def build_item(i: int, row: dict) -> None:
        item_path = OUT / "items" / f"{i:03d}.json"
        if item_path.exists():
            return
        texts = {"gpt-5.6": row["ref"]}
        for system in SYSTEMS:
            texts[system] = await rewrite(system, row)
        keys = sorted(texts)
        random.Random(f"{SEED}:{row['id']}").shuffle(keys)
        labels = "ABCDE"
        item_path.write_text(json.dumps({
            "item_id": row["id"], "source": row["source"],
            "labels": {labels[j]: {"key": k, "text": texts[k]} for j, k in enumerate(keys)},
        }, indent=1))
        (OUT / "blind" / f"{i:03d}.json").write_text(json.dumps({
            "source": row["source"],
            "candidates": {labels[j]: texts[k] for j, k in enumerate(keys)},
        }, indent=1))

    await asyncio.gather(*(build_item(i, r) for i, r in enumerate(refs)))
    print(f"items: {len(list((OUT/'items').glob('*.json')))} / blind: {len(list((OUT/'blind').glob('*.json')))}")


if __name__ == "__main__":
    asyncio.run(main())
