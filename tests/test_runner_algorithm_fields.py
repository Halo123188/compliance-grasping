"""The shared PPO cfg must not break algorithms that predate its newer fields.

``RslRlPpoAlgorithmCfg`` is one dataclass shared by every PPO-family run in this
repo, and it carries fields only a PPO SUBCLASS reads (``critic_warmup_iters``,
``std_max``). They reach the algorithm as ``**cfg["algorithm"]``, so stock
``PPO`` -- which every from-scratch teacher, every velocity run and every
tracking run uses -- died on construction with
``TypeError: unexpected keyword argument 'critic_warmup_iters'``.

The filter has to cut exactly two ways, which is what these tests pin:

  * an algorithm that does NOT accept the field must not receive it, or every
    task inheriting the shared cfg crashes;
  * an algorithm that DOES accept it must still receive it, or the fine-tune
    runs silently lose the feature they were configured for -- a failure with no
    error message, only a worse policy.

These are unit tests on the filter rather than on a constructed runner, because
the bug is entirely in which keys survive into ``train_cfg["algorithm"]``. The
filter works IN PLACE, which is also pinned here: rsl_rl mutates that same dict
(mjlab issue #764), so handing it a copy would quietly break that contract.
"""

from dataclasses import asdict

from rsl_rl.algorithms import PPO

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.rl.runner import (
  _SUBCLASS_ONLY_ALGORITHM_FIELDS,
  drop_unsupported_algorithm_fields,
)


def _cfg(class_name: str = "PPO", **overrides) -> dict:
  """A real runner cfg dict, the same shape the runners are handed."""
  train_cfg = asdict(RslRlOnPolicyRunnerCfg())
  train_cfg["algorithm"]["class_name"] = class_name
  train_cfg["algorithm"].update(overrides)
  return train_cfg


def test_shared_cfg_actually_carries_the_subclass_only_fields():
  """Guard the premise: if these ever leave the cfg, the rest is vacuous."""
  algorithm = asdict(RslRlOnPolicyRunnerCfg())["algorithm"]
  for field in _SUBCLASS_ONLY_ALGORITHM_FIELDS:
    assert field in algorithm


def test_stock_ppo_does_not_receive_subclass_only_fields():
  filtered = drop_unsupported_algorithm_fields(_cfg())["algorithm"]
  for field in _SUBCLASS_ONLY_ALGORITHM_FIELDS:
    assert field not in filtered


def test_stock_ppo_can_be_constructed_from_the_filtered_cfg():
  """The end the filter exists for: no TypeError on the real signature."""
  import inspect

  filtered = drop_unsupported_algorithm_fields(_cfg())["algorithm"]
  accepted = set(inspect.signature(PPO.__init__).parameters)
  # `class_name` is consumed by rsl_rl's construct_algorithm, not by __init__.
  # `share_cnn_encoders` is popped there too -- see the note in the filter.
  passed = set(filtered) - {"class_name", "share_cnn_encoders"}
  assert passed <= accepted, f"stock PPO would still reject {passed - accepted}"


def test_a_subclass_that_accepts_them_still_receives_them():
  """The other direction: filtering must not disable the fine-tune feature."""
  cfg = _cfg(
    class_name="mjlab.tasks.manipulation.rl.finetune:FinetunePPO",
    critic_warmup_iters=100,
    std_max=0.02,
  )
  filtered = drop_unsupported_algorithm_fields(cfg)["algorithm"]
  assert filtered["critic_warmup_iters"] == 100
  assert filtered["std_max"] == 0.02


def test_share_cnn_encoders_is_never_dropped():
  """rsl_rl pops it before __init__, so dropping it silently unshares the CNN."""
  filtered = drop_unsupported_algorithm_fields(_cfg())["algorithm"]
  assert "share_cnn_encoders" in filtered


def test_an_unresolvable_class_name_is_left_alone():
  """A cfg we cannot introspect is passed through, not silently emptied."""
  cfg = _cfg(class_name="not.a.real:Algorithm")
  before = {**cfg["algorithm"]}
  assert drop_unsupported_algorithm_fields(cfg) is cfg
  assert cfg["algorithm"] == before


def test_the_filter_works_in_place():
  """The caller's dict is the one the runner works on -- see mjlab #764."""
  cfg = _cfg()
  algorithm = cfg["algorithm"]
  returned = drop_unsupported_algorithm_fields(cfg)
  assert returned is cfg
  assert returned["algorithm"] is algorithm
  for field in _SUBCLASS_ONLY_ALGORITHM_FIELDS:
    assert field not in algorithm
