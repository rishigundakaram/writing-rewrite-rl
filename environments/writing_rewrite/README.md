# writing-rewrite

Single-turn RL environment: rewrite a messy first-draft passage (real drafts
mined from IteraTeR) into plain English per rules distilled from Zinsser's
*On Writing Well*.

**Reward** (one verifiers `GroupRewardFunc`):

```
reward_i = rank_reward_i × preservation_gate_i
```

- **Clarity rank**: one LLM-judge call sees all G rewrites of the same source
  and ranks them against the rules (`rules.txt`); rank → `(G-1-rank)/(G-1)`.
  Within-group ranking is GRPO's native signal and avoids pointwise judge
  saturation.
- **Preservation gate**: a second judge audits each rewrite against the
  source; any meaning change → 0, omissions/additions decay `0.5^n`. This
  multiplicative gate blocks the classic hack of deleting hard sentences —
  in smoke tests a claims-deleted rewrite ranks best on clarity but earns
  near-zero.

Judges run on OpenAI (`gpt-5-mini` by default, `JUDGE_MODEL` to override) —
`OPENAI_API_KEY` must be present at rollout time (`prime train ... -e OPENAI_API_KEY`).

```bash
prime env install rishigundakaram/writing-rewrite
prime eval run writing-rewrite -m openai/gpt-5-nano   # test the reward, no training
prime train configs/rl/writing.hosted.toml -e OPENAI_API_KEY
```
