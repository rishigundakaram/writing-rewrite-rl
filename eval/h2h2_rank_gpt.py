"""GPT judge pass for head-to-head round 2: rank each blind 5-way item.

One gpt-5-mini call per item (blind labels only, same criteria as the Claude
judge prompt), writing outputs/h2h2/rankings_gpt/NNN.json in the same
{"ranking": [...]} format so the same scorer works for both judges.
Idempotent per item. Needs OPENAI_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import re

from openai import AsyncOpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
BLIND = ROOT / "outputs" / "h2h2" / "blind"
OUT = ROOT / "outputs" / "h2h2" / "rankings_gpt"
JUDGE = "gpt-5-mini"

PROMPT = """\
You are a writing-quality judge. Below is a messy source draft and {n} \
candidate rewrites labeled {labels}. Rank the candidates from best to worst \
rewrite of the source. A good rewrite: (1) preserves every claim, fact, \
qualification, and negation of the source; content changes or inventions are \
disqualifying, rank them last; (2) is plain, clear English: clutter cut, \
active verbs, no cliches, no stock AI phrasing, varied sentence rhythm; \
(3) contains ONLY the rewritten text, no preambles or trailing explanations; \
(4) reads like a human wrote it.

SOURCE:
{source}

{candidates}

Answer with ONLY a JSON object: {{"ranking": ["<best label>", ..., "<worst label>"]}} \
— a permutation of {labels}.
"""


async def rank_item(client: AsyncOpenAI, path: pathlib.Path) -> None:
    out_path = OUT / path.name
    if out_path.exists():
        return
    item = json.loads(path.read_text())
    labels = sorted(item["candidates"])
    candidates = "\n\n".join(f"REWRITE {l}:\n{item['candidates'][l]}" for l in labels)
    for _ in range(3):
        response = await client.chat.completions.create(
            model=JUDGE,
            messages=[{"role": "user", "content": PROMPT.format(
                n=len(labels), labels=", ".join(labels), source=item["source"], candidates=candidates)}],
            max_completion_tokens=2500,
            reasoning_effort="low",
        )
        text = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                ranking = json.loads(match.group(0))["ranking"]
                if sorted(ranking) == labels:
                    out_path.write_text(json.dumps({"ranking": ranking}))
                    return
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"UNRANKABLE after retries: {path.name}")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(8)

    async def limited(p):
        async with sem:
            await rank_item(client, p)

    paths = sorted(BLIND.glob("*.json"))
    await asyncio.gather(*(limited(p) for p in paths))
    print(f"gpt rankings: {len(list(OUT.glob('*.json')))}/{len(paths)}")


if __name__ == "__main__":
    asyncio.run(main())
