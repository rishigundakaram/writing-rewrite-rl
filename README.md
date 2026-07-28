# writing-rewrite-rl

Post-training **Llama-3.2-3B-Instruct** with **GRPO** (via [Prime Intellect](https://www.primeintellect.ai/) `prime-rl` and the [`verifiers`](https://github.com/willccbb/verifiers) framework) to rewrite messy first-draft passages into plain English. The drafts are real revision-stage passages mined from IteraTeR; the target style is a compact rulebook derived from Zinsser's plain-English principles. The experiment compares three LLM-as-judge reward designs on the same task, each combined with a multiplicative content-preservation gate so that "delete the hard sentences" never pays.

📝 **Full write-up:** [LLMs that actually write well](https://app.notion.com/p/rishigundakaram/LLMs-that-actually-write-well-a4d94d08b6b8403f944762718349a339)

## Summary

Can a small model learn to *rewrite* a rough draft into cleaner prose? Starting from an off-the-shelf 3B model, I gave it drafts, sampled rewrites, and scored them with an LLM-as-judge reward, then trained with GRPO. The headline finding is that **reward design decided everything** — same base model, data, and compute, opposite outcomes:

- **Ranking-based rewards worked.** Rewrites from the ranking-trained models were preferred over the untrained base model **~99% of the time** in blind evaluation.
- **Naive absolute scoring got hacked.** Rewarding absolute per-attribute scores let the model game the objective — its training reward climbed to ~0.9 while the policy *degraded below baseline* (preferred over base only 26% of the time), producing short, generic text that dropped the source's meaning. The multiplicative preservation gate is what keeps the ranking rewards honest.
- **Still short of the frontier.** The best 3B models land a clear second behind GPT-5.6 (preferred over it only ~6–8%), so the training produced a large, reliable lift from the starting point without closing the gap to a frontier model.

Evaluation was blind and cross-family: rewrites were ranked by frontier judges from **OpenAI, Google, and Anthropic**, with a paper-level train/test split to prevent contamination and a separate content-preservation metric. A consistent lesson: judge *strength* matters — weaker judges flattered the small models, stronger ones did not.

## Reward designs

All three run inside one `verifiers` environment (`environments/writing_rewrite/`) and are selected with the `REWARD_MODE` env var. Every design multiplies its clarity signal by a per-completion **preservation gate**: a judge audits the rewrite against the source; any meaning change scores 0, and each omission or addition halves the gate (`0.5^n`). Formatting and em-dash penalties apply on top.

1. **Absolute attribute scoring** (`REWARD_MODE=ensemble`) — one small pointwise judge call per rule pillar, in parallel, each counting violations; the score is an inverted violation density times the gate.
2. **Group ranking** (`REWARD_MODE=rank`, default) — a single judge call sees all G rollouts of the same source and ranks them against the full rulebook; rank maps to `(G-1-rank)/(G-1)`. Within-group ranking is GRPO's native signal and avoids pointwise score saturation.
3. **Attribute-level group ranking** (`REWARD_MODE=group_parallel`) — one ranking judge per rule dimension, run in parallel over the group; per-dimension ranks are aggregated with configurable weights (`JUDGE_WEIGHTS`).

A fourth mode (`REWARD_MODE=distilled`) scores rewrites with a separately served distilled judge model (`JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`).

## Repo layout

- `environments/writing_rewrite/` — the Prime/`verifiers` single-turn environment: reward functions, judge prompts, `rules.txt`, and the train/val prompt JSONLs shipped as package data.
- `configs/rl/writing.hosted.toml` — hosted GRPO training config (400 steps, batch 64, group size 8, online win-rate eval every 40 steps).
- `pipeline/` — dataset construction: `build_prompt_dataset.py` mines and filters draft passages; `build_winrate_cache.py` pre-generates reference rewrites for online win-rate evaluation.
- `eval/` — offline evaluation: `h2h2_generate.py` produces checkpoint completions; the `h2h2_rank_*.py` scripts run blind head-to-head judging with GPT, Gemini, and Claude judges; `checkpoint_winrates.py` sweeps checkpoints; `preservation_rate.py` audits content preservation; `make_curve_images.py` renders training curves.
- `tests/` — unit tests for the deterministic parts of the reward (rank aggregation, win-rate flip, penalties).

## Setup

Python 3.10+. With `uv` (or `pip`):

```bash
uv pip install verifiers openai datasets anthropic google-auth matplotlib
```

## Running

Train on Prime Intellect (hosted):

```bash
prime env install rishigundakaram/writing-rewrite
prime eval run writing-rewrite -m openai/gpt-5-nano        # smoke-test the reward, no training
prime train configs/rl/writing.hosted.toml -e OPENAI_API_KEY -e REWARD_MODE=rank
```

Set `REWARD_MODE` to `rank`, `ensemble`, or `group_parallel` to pick the reward design; `JUDGE_MODEL` overrides the default judge (`gpt-5-mini`).

Evaluate checkpoints head-to-head:

```bash
python eval/h2h2_generate.py          # generate completions per checkpoint
python eval/h2h2_rank_gpt.py          # blind pairwise judging (also: _gemini, _claude_api, ...)
python eval/checkpoint_winrates.py    # win-rate curve across checkpoints
python eval/preservation_rate.py      # content-preservation audit
```

Tests: `python -m pytest tests/`.

## Credentials

No API keys are stored in this repo. Scripts read credentials from the environment or standard local paths:

- `OPENAI_API_KEY` — env var, used by the judge and the GPT/Gemini eval scripts.
- `~/.anthropic_key` — plain-text key file read by the Claude eval scripts.
- `~/.prime/config.json` — Prime CLI auth (created by `prime login`).
- Google Application Default Credentials — used by the Vertex (Gemini) judges.

## License

MIT — see [LICENSE](LICENSE).
