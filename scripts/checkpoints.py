from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
LATEST_CHECKPOINT_MANIFEST = 'latest_checkpoint.json'
TAGGED_CHECKPOINT_MANIFEST = 'checkpoint_ep_{episode_tag}.json'
ACTOR_TAG_PATTERN = re.compile('actor_agent_\\d+_ep_(\\d+)\\.pth$')
CRITIC_TAG_PATTERN = re.compile('critic_agent_\\d+_ep_(\\d+)\\.pth$')
RUN_TOKEN_PATTERN = re.compile('[^a-z0-9]+')

def slugify_checkpoint_token(value: str | None, *, default: str) -> str:
    text = str(value or '').strip().lower()
    text = RUN_TOKEN_PATTERN.sub('_', text).strip('_')
    return text or default

def build_checkpoint_budget_token(*, train_episodes: int | None, max_train_steps: int | None) -> str:
    if max_train_steps is not None:
        return f'steps{int(max_train_steps)}'
    return f'ep{int(train_episodes or 0)}'

def _format_run_timestamp(timestamp: datetime | str | None=None) -> str:
    if timestamp is None:
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    if isinstance(timestamp, datetime):
        return timestamp.strftime('%Y%m%d_%H%M%S')
    return str(timestamp).strip()

def build_training_run_label(*, algorithm: str, prediction_mode: str, experiment_name: str, train_episodes: int | None, max_train_steps: int | None, timestamp: datetime | str | None=None) -> str:
    algorithm_token = slugify_checkpoint_token(algorithm, default='model')
    prediction_token = slugify_checkpoint_token(prediction_mode, default='perfect')
    experiment_token = slugify_checkpoint_token(experiment_name, default='grid_mainline')
    budget_token = build_checkpoint_budget_token(train_episodes=train_episodes, max_train_steps=max_train_steps)
    time_token = slugify_checkpoint_token(_format_run_timestamp(timestamp), default='run')
    return '_'.join([algorithm_token, prediction_token, experiment_token, budget_token, time_token])

def build_training_run_paths(checkpoint_root, *, algorithm: str, prediction_mode: str, experiment_name: str, train_episodes: int | None, max_train_steps: int | None, timestamp: datetime | str | None=None) -> dict[str, object]:
    checkpoint_root = Path(checkpoint_root).resolve()
    algorithm_token = str(algorithm).strip()
    prediction_token = slugify_checkpoint_token(prediction_mode, default='perfect')
    experiment_token = slugify_checkpoint_token(experiment_name, default='grid_mainline')
    run_label = build_training_run_label(algorithm=algorithm_token, prediction_mode=prediction_token, experiment_name=experiment_token, train_episodes=train_episodes, max_train_steps=max_train_steps, timestamp=timestamp)
    model_root = checkpoint_root / algorithm_token / prediction_token / experiment_token / run_label
    meta_dir = model_root / '_meta'
    return {'checkpoint_root': checkpoint_root, 'algorithm': algorithm_token, 'prediction_mode': prediction_token, 'experiment_name': experiment_token, 'run_label': run_label, 'model_root': model_root, 'meta_dir': meta_dir, 'result_json_path': meta_dir / 'train_result.json', 'progress_json_path': meta_dir / 'progress.json', 'log_path': meta_dir / 'train.log'}

def find_latest_training_run(checkpoint_root, *, algorithm: str, prediction_mode: str, experiment_name: str) -> Path:
    checkpoint_root = Path(checkpoint_root).resolve()
    base_dir = checkpoint_root / str(algorithm).strip() / slugify_checkpoint_token(prediction_mode, default='perfect') / slugify_checkpoint_token(experiment_name, default='grid_mainline')
    if not base_dir.exists():
        raise FileNotFoundError(f"No training runs found under '{base_dir}'.")

    def _run_sort_key(path: Path) -> tuple[str, str]:
        parts = path.name.rsplit('_', 2)
        timestamp_key = '_'.join(parts[-2:]) if len(parts) >= 3 else path.name
        return (timestamp_key, path.name)
    candidates = sorted((path for path in base_dir.iterdir() if path.is_dir() and path.name != '_meta'), key=_run_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No run directories found under '{base_dir}'.")
    return candidates[-1]

def get_algorithm_checkpoint_dir(model_dir, algorithm: str) -> Path:
    return Path(model_dir) / algorithm

def checkpoint_tag_exists(algo_dir, episode_tag: int) -> bool:
    algo_dir = Path(algo_dir)
    actor_files = list(algo_dir.glob(f'actor_agent_*_ep_{episode_tag}.pth'))
    critic_files = list(algo_dir.glob(f'critic_agent_*_ep_{episode_tag}.pth'))
    return bool(actor_files) and bool(critic_files)

def infer_latest_checkpoint_tag(model_dir, algorithm: str) -> int:
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    if not algo_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: '{algo_dir}'")
    actor_tags = {int(match.group(1)) for path in algo_dir.glob('actor_agent_*_ep_*.pth') if (match := ACTOR_TAG_PATTERN.match(path.name)) is not None}
    critic_tags = {int(match.group(1)) for path in algo_dir.glob('critic_agent_*_ep_*.pth') if (match := CRITIC_TAG_PATTERN.match(path.name)) is not None}
    common_tags = sorted(actor_tags & critic_tags)
    if not common_tags:
        raise FileNotFoundError(f"No complete checkpoint tags were found under '{algo_dir}'.")
    return common_tags[-1]

def build_checkpoint_manifest(*, algorithm: str, saved_episode_tag: int, episodes_completed: int, total_steps: int, num_envs: int, episode_limit: int, save_dir) -> dict:
    return {'algorithm': algorithm, 'saved_episode_tag': int(saved_episode_tag), 'episodes_completed': int(episodes_completed), 'total_steps': int(total_steps), 'num_envs': int(num_envs), 'episode_limit': int(episode_limit), 'save_dir': str(Path(save_dir).resolve())}

def write_checkpoint_manifest(algo_dir, manifest: dict) -> dict:
    algo_dir = Path(algo_dir)
    algo_dir.mkdir(parents=True, exist_ok=True)
    latest_path = algo_dir / LATEST_CHECKPOINT_MANIFEST
    tagged_path = algo_dir / TAGGED_CHECKPOINT_MANIFEST.format(episode_tag=manifest['saved_episode_tag'])
    payload = dict(manifest)
    latest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tagged_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload

def load_latest_checkpoint_manifest(model_dir, algorithm: str) -> dict:
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    manifest_path = algo_dir / LATEST_CHECKPOINT_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Latest checkpoint manifest not found: '{manifest_path}'")
    return json.loads(manifest_path.read_text(encoding='utf-8'))

def resolve_checkpoint_to_load(model_dir, algorithm: str, episode_tag: int | None=None) -> dict:
    algo_dir = get_algorithm_checkpoint_dir(model_dir, algorithm)
    manifest = None
    if episode_tag is None:
        try:
            manifest = load_latest_checkpoint_manifest(model_dir, algorithm)
            candidate_tag = int(manifest['saved_episode_tag'])
            if checkpoint_tag_exists(algo_dir, candidate_tag):
                episode_tag = candidate_tag
        except FileNotFoundError:
            manifest = None
    if episode_tag is None:
        episode_tag = infer_latest_checkpoint_tag(model_dir, algorithm)
    if not checkpoint_tag_exists(algo_dir, int(episode_tag)):
        raise FileNotFoundError(f"Checkpoint tag {episode_tag} was not found under '{algo_dir}'.")
    resolved = {'algorithm': algorithm, 'algo_dir': str(algo_dir), 'saved_episode_tag': int(episode_tag)}
    if manifest is not None:
        resolved.update({'episodes_completed': int(manifest.get('episodes_completed', episode_tag)), 'total_steps': int(manifest.get('total_steps', 0)), 'num_envs': int(manifest.get('num_envs', 0)), 'episode_limit': int(manifest.get('episode_limit', 0)), 'save_dir': manifest.get('save_dir', str(algo_dir))})
    return resolved
