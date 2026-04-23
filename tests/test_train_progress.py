import scripts.train as train_module
from scripts.builder import build_train_runner
from tests.support.helpers import make_case_dir, make_smoke_config


class DummyTqdm:
    instances = []

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
        self.unit = kwargs.get("unit")
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


def test_train_runner_batches_progress_updates_by_episode_interval(monkeypatch, tmp_path):
    case_dir = make_case_dir(tmp_path, "train_progress")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.env.episode_limit = 3
    cfg.train.train_episodes = 4
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.train.num_envs = 1
    cfg.train.progress_episode_interval = 2
    cfg.train.show_progress = True
    cfg.train.use_noise_decay = False

    DummyTqdm.instances.clear()
    monkeypatch.setattr(train_module, "tqdm", DummyTqdm)

    runner = build_train_runner(cfg, seed=0, env_name="ProgressTest", number=1)
    try:
        runner.run()
    finally:
        runner.close()

    progress = DummyTqdm.instances[-1]
    assert progress.total == 12
    assert progress.unit == "step"
    assert progress.update_calls == [1] * 12
    assert progress.postfix_calls == 2
    assert "eta" in progress.last_postfix

    assert len(runner.history) == 0
    assert "sample_time_s" in runner.perf_summary
    assert "history_time_s" in runner.perf_summary
    assert "agent_update_time_s" in runner.perf_summary


def test_train_runner_emits_final_progress_for_partial_episode_batch(monkeypatch, tmp_path):
    case_dir = make_case_dir(tmp_path, "train_progress_partial")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.env.episode_limit = 4
    cfg.train.max_train_steps = 5
    cfg.train.num_envs = 1
    cfg.train.progress_episode_interval = 10
    cfg.train.show_progress = True
    cfg.train.use_noise_decay = False

    DummyTqdm.instances.clear()
    monkeypatch.setattr(train_module, "tqdm", DummyTqdm)

    runner = build_train_runner(cfg, seed=0, env_name="ProgressPartialTest", number=1)
    try:
        runner.run()
    finally:
        runner.close()

    progress = DummyTqdm.instances[-1]
    assert progress.total == 5
    assert progress.unit == "step"
    assert progress.update_calls == [1] * 5
    assert progress.postfix_calls == 1
    assert "eta" in progress.last_postfix


def test_train_runner_does_not_fabricate_postfix_without_completed_episode(monkeypatch, tmp_path):
    case_dir = make_case_dir(tmp_path, "train_progress_no_episode")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.env.episode_limit = 10
    cfg.train.max_train_steps = 5
    cfg.train.num_envs = 1
    cfg.train.progress_episode_interval = 10
    cfg.train.show_progress = True
    cfg.train.use_noise_decay = False

    DummyTqdm.instances.clear()
    monkeypatch.setattr(train_module, "tqdm", DummyTqdm)

    runner = build_train_runner(cfg, seed=0, env_name="ProgressNoEpisodeTest", number=1)
    try:
        runner.run()
    finally:
        runner.close()

    progress = DummyTqdm.instances[-1]
    assert progress.total == 5
    assert progress.unit == "step"
    assert progress.update_calls == [1] * 5
    assert progress.postfix_calls == 0


def test_train_runner_updates_progress_before_postfix_threshold_with_parallel_envs(monkeypatch, tmp_path):
    case_dir = make_case_dir(tmp_path, "train_progress_parallel")
    cfg = make_smoke_config(case_dir, algorithm="MADDPG")
    cfg.env.episode_limit = 3
    cfg.train.train_episodes = 8
    cfg.train.max_train_steps = cfg.train.train_episodes * cfg.env.episode_limit
    cfg.train.num_envs = 4
    cfg.train.vec_env_type = "dummy"
    cfg.train.progress_episode_interval = 10
    cfg.train.show_progress = True
    cfg.train.use_noise_decay = False

    DummyTqdm.instances.clear()
    monkeypatch.setattr(train_module, "tqdm", DummyTqdm)

    runner = build_train_runner(cfg, seed=0, env_name="ProgressParallelTest", number=1)
    try:
        runner.run()
    finally:
        runner.close()

    progress = DummyTqdm.instances[-1]
    assert progress.total == 6
    assert progress.unit == "step"
    assert progress.update_calls == [1] * 6
    assert progress.postfix_calls == 1
    assert "eta" in progress.last_postfix
