import json

import scripts.train as train_module
from scripts.builder import build_train_runner
from tests.support.helpers import make_case_dir, make_smoke_config


class DummyTqdm:
    instances = []

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
        self.disable = kwargs.get("disable", False)
        self.update_calls: list[int] = []
        self.postfix_calls = 0
        DummyTqdm.instances.append(self)

    def update(self, value=1):
        self.update_calls.append(int(value))

    def set_postfix(self, payload):
        self.postfix_calls += 1
        self.last_postfix = dict(payload)

    def close(self):
        return None


def test_train_runner_batches_progress_updates(monkeypatch, tmp_path):
    case_dir = make_case_dir(tmp_path, "train_progress")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.train.max_train_steps = 5
    cfg.train.num_envs = 1
    cfg.train.progress_postfix_interval = 2
    cfg.train.show_progress = True
    cfg.train.use_noise_decay = False
    progress_path = case_dir / "progress.json"
    cfg.runtime.progress_state_path = str(progress_path)

    DummyTqdm.instances.clear()
    monkeypatch.setattr(train_module, "tqdm", DummyTqdm)

    runner = build_train_runner(cfg, seed=0, env_name="ProgressTest", number=1)
    try:
        runner.run()
    finally:
        runner.close()

    progress = DummyTqdm.instances[-1]
    assert progress.update_calls == [2, 2, 1]
    assert progress.postfix_calls == 3

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["interaction_step"] == 5
    assert payload["target_interactions"] == 5
