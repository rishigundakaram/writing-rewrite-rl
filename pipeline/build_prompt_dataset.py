"""Build the RL prompt dataset from IteraTeR (real human first drafts).

Source: wanyu/IteraTeR_human_doc — 559 documents (arxiv/wikipedia/news), each
a (before_revision, after_revision) pair with span-level edits labeled by
intent (fluency/coherence/clarity/style/meaning-changed). We want the DRAFT
side of documents whose human reviser made clarity/style edits: text a human
judged to be badly written, not merely incomplete.

Three artifacts (JSONL under data/prompts/):
- train.jsonl / val.jsonl — rewrite prompts: 80-300-word passages chunked
  from draft sides on sentence boundaries, ranked by clarity/style edit
  density. Split by doc_id hash so passages of one document never straddle
  the split.
- calibration.jsonl — (draft, revised) passage pairs where both sides fit
  the length band; the judge ensemble should score revised >= draft. These
  never enter RL training.

--include-full adds wanyu/IteraTeR_full_doc (model-labeled intents, much
larger) for volume; human_doc alone yields a few hundred prompts, which is
enough for a first GRPO run.

Idempotent: skips if data/prompts/ is populated (--force to redo).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

import pydantic
from datasets import concatenate_datasets, load_dataset

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "data" / "prompts"

MIN_WORDS = 80
MAX_WORDS = 550
# Per-chunk word targets, cycled deterministically (seeded by doc id) so the
# dataset spans single-paragraph prompts to few-paragraph prompts. Greedy
# chunking cuts at the first sentence boundary past the target; without a
# target, chunks collapse to barely-over-MIN_WORDS (observed median 92).
TARGET_CYCLE = (100, 180, 260, 380, 500)
STYLE_INTENTS = frozenset({"clarity", "style"})
# passages must overlap at least this many clarity/style edits to qualify
MIN_STYLE_EDITS = 2
# IteraTeR_full_doc carries no intent labels; use edit density as the quality
# proxy there, and cap how many of its docs we take so the human-labeled set
# stays the anchor of the distribution.
MIN_UNLABELED_EDITS = 4
MAX_FULL_DOCS = 4000
VAL_FRACTION = 0.2


class PromptRecord(pydantic.BaseModel):
    id: str
    text: str
    domain: str
    doc_id: str
    n_style_edits: int
    n_words: int


class CalibrationRecord(pydantic.BaseModel):
    id: str
    draft: str
    revised: str
    domain: str
    doc_id: str
    n_style_edits: int


def n_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def style_edit_positions(edit_actions: list[dict]) -> list[int]:
    return [
        e["start_char_pos"]
        for e in edit_actions
        if e.get("major_intent") in STYLE_INTENTS and e.get("start_char_pos") is not None
    ]


def chunk_passages(
    text: str, sent_pos: list[int], style_pos: list[int], doc_key: str
) -> list[tuple[str, int]]:
    """Greedy sentence-boundary chunks with varied per-chunk word targets,
    plus their clarity/style edit counts. Non-overlapping, in document order.
    The target cycle's phase is seeded by doc_key so the length mix is stable
    across runs but not aligned across documents."""
    bounds = [p for p in sent_pos if 0 <= p <= len(text)] + [len(text)]
    chunks: list[tuple[str, int]] = []
    start_idx = 0
    phase = int(hashlib.sha256(doc_key.encode()).hexdigest()[:4], 16)
    while start_idx < len(bounds) - 1:
        target = TARGET_CYCLE[(phase + len(chunks)) % len(TARGET_CYCLE)]
        end_idx = start_idx + 1
        while end_idx < len(bounds) - 1 and n_words(text[bounds[start_idx] : bounds[end_idx]]) < target:
            next_words = n_words(text[bounds[start_idx] : bounds[end_idx + 1]])
            if next_words > MAX_WORDS:
                break
            end_idx += 1
        lo, hi = bounds[start_idx], bounds[end_idx]
        chunk = text[lo:hi].strip()
        words = n_words(chunk)
        if MIN_WORDS <= words <= MAX_WORDS:
            n_style = sum(1 for p in style_pos if lo <= p < hi)
            chunks.append((chunk, n_style))
        start_idx = end_idx
    return chunks


def split_of_doc(doc_id: str) -> str:
    """Split by the UNDERLYING PAPER id: strip the source-corpus prefix and
    revision depth, so all revisions of one paper (heavily overlapping text)
    land on the same side, and the same paper appearing in both the human and
    full corpora cannot straddle. The old per-doc_id hash leaked 196 papers
    across the split."""
    base = re.sub(r"^iterater_\w+:", "", doc_id)
    base = re.sub(r":r\d+$", "", base)
    h = int(hashlib.sha256(base.encode()).hexdigest()[:8], 16)
    return "val" if (h % 100) < VAL_FRACTION * 100 else "train"


def build(include_full: bool) -> tuple[list[PromptRecord], list[PromptRecord], list[CalibrationRecord]]:
    sources = [("iterater_human", load_dataset("wanyu/IteraTeR_human_doc"))]
    if include_full:
        sources.append(("iterater_full", load_dataset("wanyu/IteraTeR_full_doc")))

    seen: set[str] = set()
    train: list[PromptRecord] = []
    val: list[PromptRecord] = []
    calibration: list[CalibrationRecord] = []

    for source_name, ds in sources:
        rows = concatenate_datasets([ds[s] for s in ds])
        n_full_taken = 0
        for row in rows:
            edits = row["edit_actions"]
            labeled = any("major_intent" in e for e in edits)
            if labeled:
                style_pos = style_edit_positions(edits)
                if len(style_pos) < MIN_STYLE_EDITS:
                    continue
            else:
                # unlabeled corpus: edit density proxy, capped contribution
                if len(edits) < MIN_UNLABELED_EDITS or n_full_taken >= MAX_FULL_DOCS:
                    continue
                n_full_taken += 1
                style_pos = [e["start_char_pos"] for e in edits if e.get("start_char_pos") is not None]
            doc_id = f"{source_name}:{row['doc_id']}:r{row['revision_depth']}"
            before = row["before_revision"]
            # a handful of rows carry a null domain; keep them, labeled
            domain = (row.get("domain") or "unknown") if "domain" in row else "full_unlabeled"

            # calibration pair: whole doc when both sides fit the band
            if (
                MIN_WORDS <= n_words(before) <= MAX_WORDS
                and MIN_WORDS <= n_words(row["after_revision"]) <= MAX_WORDS
            ):
                calibration.append(
                    CalibrationRecord(
                        id=f"cal:{doc_id}",
                        draft=before,
                        revised=row["after_revision"],
                        domain=domain,
                        doc_id=doc_id,
                        n_style_edits=len(style_pos),
                    )
                )

            for i, (chunk, n_style) in enumerate(
                chunk_passages(before, row["sents_char_pos"], style_pos, doc_id)
            ):
                if n_style < MIN_STYLE_EDITS:
                    continue
                key = re.sub(r"\s+", " ", chunk.lower())[:400]
                if key in seen:
                    continue
                seen.add(key)
                record = PromptRecord(
                    id=f"{doc_id}:c{i}",
                    text=chunk,
                    domain=domain,
                    doc_id=doc_id,
                    n_style_edits=n_style,
                    n_words=n_words(chunk),
                )
                (val if split_of_doc(doc_id) == "val" else train).append(record)

    return train, val, calibration


def write_jsonl(path: pathlib.Path, records: list[pydantic.BaseModel]) -> None:
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))
    print(f"wrote {path.name}: {len(records)} records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-full", action="store_true", help="add IteraTeR_full_doc for volume")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if PROMPTS_DIR.exists() and any(PROMPTS_DIR.glob("*.jsonl")) and not args.force:
        print(f"{PROMPTS_DIR} already populated, use --force")
        return

    train, val, calibration = build(args.include_full)
    if len(train) < 100:
        print(f"WARNING: only {len(train)} train prompts — consider --include-full")
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROMPTS_DIR / "train.jsonl", train)
    write_jsonl(PROMPTS_DIR / "val.jsonl", val)
    write_jsonl(PROMPTS_DIR / "calibration.jsonl", calibration)

    domains: dict[str, int] = {}
    for r in train + val:
        domains[r.domain] = domains.get(r.domain, 0) + 1
    print("domain spread:", domains)


if __name__ == "__main__":
    main()
