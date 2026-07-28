"""Build the win-rate reference cache: gpt-5.6 rewrites of clean-val prompts.

Picks 128 prompts from the NEW paper-level val split (deterministic seed,
length-stratified across the TARGET_CYCLE bands), asserts zero train
contamination at the paper level, generates one gpt-5.6 rewrite per prompt
with the exact training INSTRUCTION, and writes
environments/writing_rewrite/writing_rewrite/winrate_refs.jsonl
({id, source, ref}).

These 128 prompts + refs are the shared basis for: the in-training winrate
eval (task="winrate"), the post-hoc Claude winrate curves, and the final
blind head-to-head. Idempotent; --force to regenerate. Needs OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import re
import sys

from openai import AsyncOpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "environments" / "writing_rewrite"))

from pipeline.build_prompt_dataset import split_of_doc  # noqa: E402 - the fixed paper-level split
import writing_rewrite as wr  # noqa: E402 - INSTRUCTION

OUT = ROOT / "environments" / "writing_rewrite" / "writing_rewrite" / "winrate_refs.jsonl"
VAL = ROOT / "data" / "prompts" / "val.jsonl"
TRAIN = ROOT / "data" / "prompts" / "train.jsonl"

N_PROMPTS = 128
SEED = 20260722
REF_MODEL = "gpt-5.6"


def base_paper(doc_id: str) -> str:
    return re.sub(r":r\d+$", "", re.sub(r"^iterater_\w+:", "", doc_id))


def pick_prompts() -> list[dict]:
    val = [json.loads(l) for l in VAL.read_text().splitlines() if l]
    train_papers = {base_paper(json.loads(l)["doc_id"]) for l in TRAIN.read_text().splitlines() if l}

    # hard assertions: the split function agrees, and no paper overlap
    for r in val:
        if split_of_doc(r["doc_id"]) != "val":
            raise AssertionError(f"{r['id']} is in val.jsonl but split_of_doc says train")
        if base_paper(r["doc_id"]) in train_papers:
            raise AssertionError(f"{r['id']} shares a paper with the train split")

    # length-stratified deterministic sample: sort into 4 length bands, take evenly
    rng = random.Random(SEED)
    bands: list[list[dict]] = [[], [], [], []]
    for r in val:
        w = r["n_words"]
        bands[0 if w < 130 else 1 if w < 220 else 2 if w < 330 else 3].append(r)
    per_band = N_PROMPTS // len(bands)
    picked: list[dict] = []
    for band in bands:
        rng.shuffle(band)
        picked.extend(band[:per_band])
    # top up from the largest bands if any band was short
    if len(picked) < N_PROMPTS:
        rest = [r for band in bands for r in band[per_band:]]
        rng.shuffle(rest)
        picked.extend(rest[: N_PROMPTS - len(picked)])
    return picked[:N_PROMPTS]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} exists, use --force")
        return

    prompts = pick_prompts()
    print(f"picked {len(prompts)} clean-val prompts (contamination assertions passed)")
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(8)

    async def ref_for(row: dict) -> dict:
        async with sem:
            for attempt in range(3):
                response = await client.chat.completions.create(
                    model=REF_MODEL,
                    messages=[{"role": "user", "content": wr.INSTRUCTION + row["text"]}],
                    max_completion_tokens=1200,
                )
                ref = (response.choices[0].message.content or "").strip()
                if ref:
                    return {"id": row["id"], "source": row["text"], "ref": ref}
        raise RuntimeError(f"empty gpt-5.6 rewrite for {row['id']} after retries")

    rows = await asyncio.gather(*(ref_for(r) for r in prompts))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {OUT.name}: {len(rows)} refs")


if __name__ == "__main__":
    asyncio.run(main())
