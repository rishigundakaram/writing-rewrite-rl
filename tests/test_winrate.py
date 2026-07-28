import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "environments" / "writing_rewrite"))

import writing_rewrite as wr


def test_flip_is_deterministic_and_mixed() -> None:
    ids = [f"iterater_full:{i}:r1:c0" for i in range(400)]
    flips = [wr._winrate_flip(i) for i in ids]
    assert flips == [wr._winrate_flip(i) for i in ids]
    frac = sum(flips) / len(flips)
    assert 0.4 < frac < 0.6  # both orders well represented


def test_winrate_reward_mapping() -> None:
    # policy in slot A
    assert wr.winrate_reward_value("A", policy_is_a=True) == 1.0
    assert wr.winrate_reward_value("B", policy_is_a=True) == 0.0
    # policy in slot B
    assert wr.winrate_reward_value("B", policy_is_a=False) == 1.0
    assert wr.winrate_reward_value("A", policy_is_a=False) == 0.0
    # unparseable verdicts carry no signal
    assert wr.winrate_reward_value(None, policy_is_a=True) == 0.5
    assert wr.winrate_reward_value("C", policy_is_a=False) == 0.5


def test_task_dispatch_rejects_unknown() -> None:
    try:
        wr.load_environment(task="bogus")
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError")
