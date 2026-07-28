"""Plain-English rewrite RL environment (Prime Intellect / verifiers).

Task: given a messy first-draft passage (real drafts mined from IteraTeR),
rewrite it into plain English per the Zinsser-derived rules in rules.txt.

Reward = within-group clarity RANK x per-completion preservation GATE, in one
verifiers GroupRewardFunc:
- Ranking: ONE judge call sees all G rewrites of the same source and ranks
  them against the rules. Rank -> (G-1-rank)/(G-1). Within-group ranking is
  GRPO's native signal and avoids pointwise score saturation.
- Gate: a preservation judge scores each rewrite against the source;
  meaning_change -> 0, omissions/additions decay 0.5^n. Multiplicative, so a
  top-ranked rewrite that drops content still earns ~nothing ("delete the
  hard sentences" must never pay).

Self-contained on purpose: no imports from the sibling writing_env package,
plain OpenAI structured-output calls, rules.txt + prompt JSONLs shipped as
package data — so `prime env push` works. Needs OPENAI_API_KEY at rollout
time (judge calls). JUDGE_MODEL env var overrides the default gpt-5-mini.

REWARD_MODE=distilled switches to the pointwise distilled-judge reward: each
rewrite is scored by OUR RL-trained judge (the writing-judge adapter served
on Prime Inference) instead of frontier-judge ranking+gating. The distilled
judge predicts violations + a preservation-defect kind in one call;
    score = 1 / (1 + violations per 100 words) x gate_factor(kind).
Configure with JUDGE_BASE_URL (e.g. https://api.pinference.ai/api/v1),
JUDGE_API_KEY, and JUDGE_MODEL=<adapter id>. This is the A/B arm for
"can a distilled judge replace a frontier judge as an RL reward".

Smoke-test the reward without training:
    prime env install writing-rewrite
    prime eval run writing-rewrite -m openai/gpt-5-nano
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re

import pydantic
import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

_DIR = pathlib.Path(__file__).parent
RULES = (_DIR / "rules.txt").read_text()

DEFAULT_JUDGE_MODEL = "gpt-5-mini"

INSTRUCTION = (
    "Rewrite the passage below into plain English. Cut clutter, prefer plain "
    "words and active verbs, drop cliches and stock phrasing, keep a human "
    "voice. Preserve every claim, fact, qualification, and negation exactly; "
    "add nothing. Reply with ONLY the rewritten passage.\n\nPASSAGE:\n"
)

RANK_PROMPT = """\
You judge rewrites of a messy source passage against these rules:

{rules}

SOURCE PASSAGE:
{source}

Below are {n} candidate rewrites, numbered 0 to {n_minus_1}. Rank them from
best to worst by how well they follow the rules: fewest violations wins;
between two clean rewrites prefer the one with more natural, varied, human
prose. Judge ONLY writing quality per the rules — do NOT judge content
fidelity to the source (a separate judge handles that).

{candidates}

Return JSON: {{"ranking": [best index, ..., worst index]}} — a permutation of
0..{n_minus_1}. Add a one-sentence "rationale".
"""

PRESERVE_PROMPT = """\
Does the REWRITE preserve the meaning of the SOURCE? The rewrite may cut
clutter and restructure freely; it may not change what the text says.

List every issue:
- omission: a claim/fact/qualification/negation in the source is missing
  (cutting redundant restatement is NOT an omission)
- addition: the rewrite asserts something material the source does not
- meaning_change: a claim survives but says something different (dropped
  "not", some->all, hedge stated as fact)

Zero issues is the expected answer for a faithful rewrite. Check
claim-by-claim: numbers, exceptions, negations, attributions.

SOURCE:
{source}

REWRITE:
{rewrite}

Return JSON: {{"issues": [{{"type": "omission|addition|meaning_change", "detail": "..."}}]}}
"""


class RankVerdict(pydantic.BaseModel):
    ranking: list[int]
    rationale: str


class PreserveIssue(pydantic.BaseModel):
    type: str
    detail: str


class PreserveVerdict(pydantic.BaseModel):
    issues: list[PreserveIssue]


def _client() -> AsyncOpenAI:
    return AsyncOpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL


def _judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def _len_kwargs(model: str, n: int) -> dict:
    """OpenAI gpt-5* models take max_completion_tokens (which the model's
    hidden reasoning also consumes — give headroom and cap effort, or short
    caps yield empty content and parse-neutral rewards); everything else
    (pinference adapters, etc.) takes max_tokens."""
    if model.startswith("gpt-"):
        return {"max_completion_tokens": n + 2000, "reasoning_effort": "low"}
    return {"max_tokens": n}


def _reward_mode() -> str:
    mode = os.environ.get("REWARD_MODE", "rank")
    if mode not in ("rank", "distilled", "ensemble", "group_parallel"):
        raise ValueError(
            f"REWARD_MODE={mode!r}; use 'rank', 'distilled', 'ensemble', or 'group_parallel'"
        )
    return mode


def _distilled_client() -> AsyncOpenAI:
    base_url = os.environ.get("JUDGE_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise ValueError("REWARD_MODE=distilled requires JUDGE_BASE_URL and JUDGE_API_KEY")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# Mirrors the writing-judge v2 training prompt exactly (count-per-pillar
# task) so the distilled judge sees its own training distribution
# (TEXT = the candidate rewrite).
RULES_COMPACT = (_DIR / "rules_compact.txt").read_text()

DISTILLED_PILLARS = (
    "plainness_and_decluttering",
    "verb_power_and_active_structure",
    "freshness_not_cliche",
    "momentum_leads_endings",
    "anti_llm_slop",
)

DISTILLED_JUDGE_PROMPT = """\
You are a writing judge. The TEXT below is a corrupted version of the SOURCE:
style violations were planted in it, and its meaning may have been quietly
changed.

The style rubric (5 pillars; the bullet points show what counts as one
violation of that pillar):

{rules}

Count the violations of each pillar in TEXT, and judge whether TEXT preserves
the meaning of SOURCE. Answer with ONLY a JSON object:
{{"counts": {{"plainness_and_decluttering": <int>, "verb_power_and_active_structure": <int>, "freshness_not_cliche": <int>, "momentum_leads_endings": <int>, "anti_llm_slop": <int>}},
 "gate": "none" | "subtle_omission" | "meaning_inversion" | "addition"}}

"gate": subtle_omission = a material claim/qualification was dropped;
meaning_inversion = a claim now says something different; addition = a
material claim was invented; none = meaning intact.

SOURCE:
{source}

TEXT:
{text}
"""

_GATE_FACTOR = {"none": 1.0, "subtle_omission": 0.5, "addition": 0.5, "meaning_inversion": 0.25}


async def _distilled_score(client: AsyncOpenAI, source: str, rewrite: str) -> float:
    """Pointwise score from the distilled judge; parse failures are neutral
    (0.5) so a flaky judge response doesn't inject random advantage."""
    if not rewrite.strip():
        return 0.0
    response = await client.chat.completions.create(
        model=_judge_model(),
        messages=[
            {
                "role": "user",
                "content": DISTILLED_JUDGE_PROMPT.format(rules=RULES_COMPACT, source=source, text=rewrite),
            }
        ],
        **_len_kwargs(_judge_model(), 400),
    )
    text = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return 0.5
    try:
        obj = json.loads(match.group(0))
        counts = obj.get("counts")
        gate_kind = obj.get("gate")
        if not isinstance(counts, dict) or gate_kind not in _GATE_FACTOR:
            return 0.5
        n_violations = sum(
            int(counts.get(p, 0)) for p in DISTILLED_PILLARS
            if isinstance(counts.get(p, 0), int) and not isinstance(counts.get(p, 0), bool) and counts.get(p, 0) >= 0
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.5
    n_words = len(rewrite.split())
    per_100 = 100.0 * n_violations / n_words if n_words else 100.0
    return (1.0 / (1.0 + per_100)) * _GATE_FACTOR[gate_kind]




# Chat-habit wrapper detection: preambles ("Here's a rewritten version:"),
# trailing self-review ("I've made the following changes..."). Deterministic
# and multiplicative, because shared habits are invisible to within-group
# ranking — every rollout having the same wrapper means no relative penalty.
_WRAPPER_RES = (
    re.compile(r"^\s*(here('s| is)|sure|certainly|below is|i('ve| have) rewritten)\b", re.IGNORECASE),
    re.compile(r"\b(i('ve| have) made|changes i made|to improve its clarity|rewritten version of)\b", re.IGNORECASE),
    re.compile(r"^\s*(note|explanation of changes|key changes)\s*:", re.IGNORECASE | re.MULTILINE),
)


def format_penalty(text: str) -> float:
    """1.0 for a bare rewrite, 0.25 when chat-wrapper habits are detected."""
    return 0.25 if any(r.search(text) for r in _WRAPPER_RES) else 1.0


def _completion_text(completion: object) -> str:
    """Final assistant text; aborted/empty rollouts yield '' and score 0.

    Reasoning models (Qwen3.5, gpt-oss, ...) emit chain-of-thought into the
    completion. Judges must see only the final rewrite, so strip closed
    <think> blocks; an UNCLOSED think block means the rollout burned its
    token budget reasoning and never answered — that scores 0, on purpose.
    """
    if isinstance(completion, list):
        text = completion[-1].get("content", "") if completion else ""
    else:
        text = str(completion)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        return ""
    return text.strip()


async def _rank(
    client: AsyncOpenAI, source: str, texts: list[str], rules: str | None = None
) -> list[int]:
    """Rank position per candidate (0 = best). Non-permutation output retries
    once, then falls back to all-equal (rank signal off, gate still applies).
    `rules` narrows the judge to one dimension (group_parallel mode); None
    uses the full rulebook (rank mode)."""
    candidates = "\n\n".join(f"[{i}]\n{t if t.strip() else '(empty rewrite)'}" for i, t in enumerate(texts))
    prompt = RANK_PROMPT.format(
        rules=rules if rules is not None else RULES,
        source=source, n=len(texts), n_minus_1=len(texts) - 1, candidates=candidates,
    )
    for _ in range(2):
        response = await client.chat.completions.parse(
            model=_judge_model(),
            messages=[{"role": "user", "content": prompt}],
            response_format=RankVerdict,
        )
        verdict = response.choices[0].message.parsed
        if verdict is not None and sorted(verdict.ranking) == list(range(len(texts))):
            positions = [0] * len(texts)
            for pos, idx in enumerate(verdict.ranking):
                positions[idx] = pos
            return positions
    return [0] * len(texts)  # degenerate: uniform rank component


def _gate_model() -> str:
    """Gates are 8x the call volume of ranking; GATE_MODEL lets them run on a
    cheaper model than the ranking judge."""
    return os.environ.get("GATE_MODEL", _judge_model())


# ---- group_parallel mode: parallel per-dimension RANKING judges ----------
# Within-group ranking proved the most stable GRPO signal, but one judge
# can't reliably hold the whole rulebook. Here each dimension gets its own
# ranking judge over the same G rewrites; normalized ranks aggregate by
# weight; the preservation gate is the only gate; deterministic checks
# (wrapper, em-dash) multiply on top.

VARIETY_RULES = """\
## sentence_variety — Vary sentence length and structure; prose must not march.
- A run of 3+ consecutive sentences with near-identical length and shape (subject-verb-object metronome) ranks a rewrite down.
- Good variety mixes short punchy sentences with longer ones, varies openings, and uses an occasional fragment or question where it earns its place.
- Do not reward variety achieved by adding filler; judge the rhythm of what is there."""

def _rank_dimensions() -> dict[str, str]:
    dims = dict(_pillar_sections())
    dims["sentence_variety"] = VARIETY_RULES
    return dims


def _dim_weights(dims: list[str]) -> dict[str, float]:
    """Uniform weights, overridable via JUDGE_WEIGHTS='{"anti_llm_slop": 2, ...}'."""
    raw = os.environ.get("JUDGE_WEIGHTS")
    overrides = json.loads(raw) if raw else {}
    unknown = set(overrides) - set(dims)
    if unknown:
        raise ValueError(f"JUDGE_WEIGHTS names unknown dimensions: {sorted(unknown)}")
    return {d: float(overrides.get(d, 1.0)) for d in dims}


def aggregate_ranks(rank_lists: list[list[int]], weights: list[float], n: int) -> list[float]:
    """Weighted mean of normalized per-dimension rank rewards, in [0, 1]."""
    total_w = sum(weights)
    out = []
    for i in range(n):
        acc = 0.0
        for positions, w in zip(rank_lists, weights):
            rank_reward = 1.0 if n == 1 else (n - 1 - positions[i]) / (n - 1)
            acc += w * rank_reward
        out.append(acc / total_w)
    return out


def emdash_penalty(source: str, rewrite: str) -> float:
    """x0.5 when the rewrite introduces em/en dashes the source didn't have —
    a signature LLM tell. Source-conditional so legitimate dash style in the
    original isn't punished."""
    if ("—" in rewrite or "–" in rewrite) and not ("—" in source or "–" in source):
        return 0.5
    return 1.0


async def _gate(client: AsyncOpenAI, source: str, rewrite: str) -> float:
    if not rewrite.strip():
        return 0.0
    response = await client.chat.completions.parse(
        model=_gate_model(),
        messages=[{"role": "user", "content": PRESERVE_PROMPT.format(source=source, rewrite=rewrite)}],
        response_format=PreserveVerdict,
    )
    verdict = response.choices[0].message.parsed
    if verdict is None:
        return 0.0
    # Exponential decay, no cliff: a hard 0 for any meaning_change removes
    # all within-group variance for weak policies (every rollout gates to 0
    # -> zero advantage -> nothing trains, observed on Llama-3.2-3B). Decay
    # keeps the ordering incentive — 1 meaning slip (0.25) beats 3 (0.016) —
    # while giving GRPO a ladder out of the dead zone.
    weighted = sum(2 if i.type == "meaning_change" else 1 for i in verdict.issues)
    return 0.5 ** weighted


def _load_rows(name: str) -> Dataset:
    rows = [json.loads(line) for line in (_DIR / name).read_text().splitlines() if line]
    return Dataset.from_dict(
        {
            "question": [INSTRUCTION + r["text"] for r in rows],
            "answer": [""] * len(rows),
            "info": [{"source": r["text"], "prompt_id": r["id"], "domain": r["domain"]} for r in rows],
        }
    )


# --- ensemble mode: one SMALL judge call per pillar, in parallel -------
# Applies the judge-diagnosis lesson at reward time: instead of one call
# holding the whole rulebook, each call sees ONE pillar's rules and counts
# only that pillar's violations. Plus one gate call. 6 small calls per
# completion, all concurrent.

def _pillar_sections() -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    for line in RULES_COMPACT.splitlines():
        if line.startswith("## "):
            current = line[3:].split(" — ")[0].strip()
            sections[current] = line + "\n"
        elif current:
            sections[current] += line + "\n"
    return sections


ENSEMBLE_PILLAR_PROMPT = """\
Count the violations of ONE writing pillar in the TEXT below.

The pillar (each bullet shows what counts as one violation):

{pillar_rules}

Answer with ONLY a JSON object: {{"count": <int>}}

TEXT:
{text}
"""

ENSEMBLE_GATE_PROMPT = """\
Does the TEXT preserve the meaning of the SOURCE? It may cut clutter and
restructure freely; it may not change what is claimed.

Answer with ONLY a JSON object:
{{"gate": "none" | "subtle_omission" | "meaning_inversion" | "addition"}}
none = meaning intact; subtle_omission = a material claim/qualification was
dropped; meaning_inversion = a claim now says something different;
addition = a material claim was invented.

SOURCE:
{source}

TEXT:
{text}
"""


async def _ensemble_pillar_count(client: AsyncOpenAI, pillar_rules: str, text: str) -> int:
    response = await client.chat.completions.create(
        model=_judge_model(),
        messages=[{"role": "user", "content": ENSEMBLE_PILLAR_PROMPT.format(pillar_rules=pillar_rules, text=text)}],
        **_len_kwargs(_judge_model(), 100),
    )
    raw = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return 1  # parse failure: prior-ish count, no directional signal
    try:
        count = json.loads(match.group(0)).get("count")
        return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 1
    except json.JSONDecodeError:
        return 1


async def _ensemble_gate(client: AsyncOpenAI, source: str, text: str) -> float:
    response = await client.chat.completions.create(
        model=_judge_model(),
        messages=[{"role": "user", "content": ENSEMBLE_GATE_PROMPT.format(source=source, text=text)}],
        **_len_kwargs(_gate_model(), 60),
    )
    raw = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return 0.75
    try:
        kind = json.loads(match.group(0)).get("gate")
    except json.JSONDecodeError:
        return 0.75
    return _GATE_FACTOR.get(kind, 0.75)


async def _ensemble_score(client: AsyncOpenAI, source: str, rewrite: str) -> float:
    if not rewrite.strip():
        return 0.0
    sections = _pillar_sections()
    counts_and_gate = await asyncio.gather(
        *(_ensemble_pillar_count(client, rules, rewrite) for rules in sections.values()),
        _ensemble_gate(client, source, rewrite),
    )
    n_violations = sum(counts_and_gate[:-1])
    gate_factor = counts_and_gate[-1]
    n_words = len(rewrite.split())
    per_100 = 100.0 * n_violations / n_words if n_words else 100.0
    return (1.0 / (1.0 + per_100)) * gate_factor


# ---- winrate eval task: pairwise policy-vs-GPT-5.6 on cached references ----
# Used as a SEPARATE eval env during hosted training ([[eval.env]] with
# args = {task = "winrate"}): mean eval reward == win rate vs gpt-5.6.

WINRATE_PROMPT = """\
Two writers rewrote the same DRAFT into plain English. Judge which rewrite is
better: fewer violations of the rules below, meaning of the draft preserved
exactly, and only the rewritten text (no preamble or commentary).

The rules, in brief:

{rules}

DRAFT:
{source}

REWRITE A:
{a}

REWRITE B:
{b}

Answer with ONLY a JSON object: {{"winner": "A" | "B"}}
"""


def _winrate_flip(row_id: str) -> bool:
    """Deterministic per-example order flip (position-bias control)."""
    import hashlib

    return int(hashlib.sha256(row_id.encode()).hexdigest()[:2], 16) % 2 == 0


def winrate_reward_value(winner: str | None, policy_is_a: bool) -> float:
    """Map the judge's verdict to reward; unparseable -> 0.5 (no signal)."""
    if winner not in ("A", "B"):
        return 0.5
    return 1.0 if (winner == "A") == policy_is_a else 0.0


async def _winrate_score(client: AsyncOpenAI, row_id: str, source: str, ref: str, rewrite: str) -> float:
    if not rewrite.strip():
        return 0.0
    policy_is_a = _winrate_flip(row_id)
    a, b = (rewrite, ref) if policy_is_a else (ref, rewrite)
    response = await client.chat.completions.create(
        model=_judge_model(),
        messages=[{"role": "user", "content": WINRATE_PROMPT.format(rules=RULES_COMPACT, source=source, a=a, b=b)}],
        **_len_kwargs(_judge_model(), 200),
    )
    text = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    winner = None
    if match:
        try:
            winner = json.loads(match.group(0)).get("winner")
        except json.JSONDecodeError:
            winner = None
    return winrate_reward_value(winner, policy_is_a)


def _load_winrate_env() -> vf.Environment:
    rows = [json.loads(l) for l in (_DIR / "winrate_refs.jsonl").read_text().splitlines() if l]
    dataset = Dataset.from_dict(
        {
            "question": [INSTRUCTION + r["source"] for r in rows],
            "answer": [""] * len(rows),
            "info": [{"source": r["source"], "ref": r["ref"], "row_id": r["id"]} for r in rows],
        }
    )

    async def winrate_vs_gpt(completion, info, **kwargs) -> float:
        return await _winrate_score(
            _client(), info["row_id"], info["source"], info["ref"], _completion_text(completion)
        )

    rubric = vf.Rubric(funcs=[winrate_vs_gpt], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, eval_dataset=dataset, rubric=rubric)


def load_environment(task: str = "train") -> vf.Environment:
    if task == "winrate":
        return _load_winrate_env()
    if task != "train":
        raise ValueError(f"task={task!r}; use 'train' or 'winrate'")
    return _load_train_environment()


def _load_train_environment() -> vf.Environment:
    dataset = _load_rows("prompts_train.jsonl")
    eval_dataset = _load_rows("prompts_val.jsonl")

    async def clarity_rank_gated(completions, infos, **kwargs) -> list[float]:
        """GroupRewardFunc: one ranking call over the group x per-completion
        preservation gates. All judge calls run concurrently."""
        texts = [_completion_text(c) for c in completions]
        if not texts:
            return []
        source = infos[0]["source"]
        client = _client()
        ranks, *gates = await asyncio.gather(
            _rank(client, source, texts),
            *(_gate(client, source, t) for t in texts),
        )
        n = len(texts)
        rank_rewards = [1.0 if n == 1 else (n - 1 - r) / (n - 1) for r in ranks]
        return [rr * g * format_penalty(t) for rr, g, t in zip(rank_rewards, gates, texts)]

    async def clarity_distilled(completions, infos, **kwargs) -> list[float]:
        """GroupRewardFunc, distilled mode: pointwise scores from our
        RL-trained judge adapter, all completions concurrently."""
        texts = [_completion_text(c) for c in completions]
        if not texts:
            return []
        source = infos[0]["source"]
        client = _distilled_client()
        scores = await asyncio.gather(*(_distilled_score(client, source, t) for t in texts))
        return [sc * format_penalty(t) for sc, t in zip(scores, texts)]

    async def clarity_ensemble(completions, infos, **kwargs) -> list[float]:
        texts = [_completion_text(c) for c in completions]
        if not texts:
            return []
        source = infos[0]["source"]
        client = _distilled_client()
        scores = await asyncio.gather(*(_ensemble_score(client, source, t) for t in texts))
        return [sc * format_penalty(t) for sc, t in zip(scores, texts)]

    async def clarity_group_parallel(completions, infos, **kwargs) -> list[float]:
        """GroupRewardFunc: 6 per-dimension ranking calls + G gate calls,
        all concurrent; weighted rank aggregation x gate x deterministic
        penalties (wrapper, em-dash)."""
        texts = [_completion_text(c) for c in completions]
        if not texts:
            return []
        source = infos[0]["source"]
        client = _client()
        dims = _rank_dimensions()
        weights = _dim_weights(list(dims))
        results = await asyncio.gather(
            *(_rank(client, source, texts, rules=rules) for rules in dims.values()),
            *(_gate(client, source, t) for t in texts),
        )
        rank_lists = list(results[: len(dims)])
        gates = list(results[len(dims):])
        agg = aggregate_ranks(rank_lists, [weights[d] for d in dims], len(texts))
        return [
            a * g * format_penalty(t) * emdash_penalty(source, t)
            for a, g, t in zip(agg, gates, texts)
        ]

    mode = _reward_mode()
    if mode in ("distilled", "ensemble"):
        _distilled_client()  # fail fast at load if JUDGE_BASE_URL/JUDGE_API_KEY missing
        reward = clarity_distilled if mode == "distilled" else clarity_ensemble
    elif mode == "group_parallel":
        _dim_weights(list(_rank_dimensions()))  # fail fast on bad JUDGE_WEIGHTS
        reward = clarity_group_parallel
    else:
        reward = clarity_rank_gated
    rubric = vf.Rubric(funcs=[reward], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, eval_dataset=eval_dataset, rubric=rubric)
