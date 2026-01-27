import os
from typing import Dict, List, Optional, Union

TASKS = [
    "text-generation",
    "image-text-to-text",
    "automatic-speech-recognition",
    "image-to-text",
    "fill-mask",
]


def list_tasks():
    return TASKS


def list_models(
    tasks: Union[str, List[str]] = TASKS,
    include_private: bool = False,
) -> Dict[str, List[str]]:
    if isinstance(tasks, str):
        tasks = [tasks]
    assert set(tasks).issubset(TASKS), f"mblt model zoo supports tasks in {TASKS}"

    try:
        return _list_models_from_hub(tasks, include_private=include_private)
    except Exception as e:
        print(
            "Failed to list models from Hugging Face Hub. "
            f"Falling back to local cache. Error: {e}"
        )
        return _list_models_from_cache(tasks)


def _list_models_from_hub(
    tasks: List[str],
    *,
    include_private: bool,
) -> Dict[str, List[str]]:
    from huggingface_hub import HfApi

    api = HfApi()
    available_models: Dict[str, List[str]] = {task: [] for task in tasks}
    for task in tasks:
        models = api.list_models(author="mobilint", pipeline_tag=task)
        ids: List[str] = []
        for model in models:
            if not include_private and bool(getattr(model, "private", False)):
                continue
            model_id = (
                getattr(model, "modelId", None)
                or getattr(model, "model_id", None)
                or getattr(model, "id", None)
            )
            if model_id and model_id.startswith("mobilint/"):
                ids.append(model_id)
        available_models[task] = ids
    return available_models


def _list_models_from_cache(tasks: List[str]) -> Dict[str, List[str]]:
    cache_root = _get_hf_cache_root()
    available_models: Dict[str, List[str]] = {task: [] for task in tasks}
    if not os.path.isdir(cache_root):
        return available_models

    unmatched: List[str] = []
    for entry in os.listdir(cache_root):
        if not entry.startswith("models--mobilint--"):
            continue
        repo_dir = os.path.join(cache_root, entry)
        model_id = _model_id_from_cache_dir(entry)
        if model_id is None:
            continue
        pipeline_tag = _read_pipeline_tag(repo_dir)
        if pipeline_tag in tasks:
            available_models[pipeline_tag].append(model_id)
        else:
            unmatched.append(model_id)

    if not unmatched:
        return available_models

    if len(tasks) == 1:
        available_models[tasks[0]].extend(unmatched)
        return available_models

    if sum(len(models) for models in available_models.values()) == 0:
        for task in tasks:
            available_models[task].extend(unmatched)

    return available_models


def _get_hf_cache_root() -> str:
    env_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
    if env_cache:
        return env_cache
    hf_home = os.getenv("HF_HOME") or os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
    )
    return os.path.join(hf_home, "hub")


def _model_id_from_cache_dir(entry: str) -> Optional[str]:
    parts = entry.split("--")
    if len(parts) < 3 or parts[0] != "models":
        return None
    org = parts[1]
    repo = "--".join(parts[2:])
    return f"{org}/{repo}"


def _read_pipeline_tag(repo_dir: str) -> Optional[str]:
    commit = _read_ref_commit(os.path.join(repo_dir, "refs", "main"))
    snapshot_dir = None
    if commit:
        candidate = os.path.join(repo_dir, "snapshots", commit)
        if os.path.isdir(candidate):
            snapshot_dir = candidate
    if snapshot_dir is None:
        snapshots_root = os.path.join(repo_dir, "snapshots")
        if os.path.isdir(snapshots_root):
            entries = [
                os.path.join(snapshots_root, name)
                for name in os.listdir(snapshots_root)
            ]
            snapshot_dirs = [p for p in entries if os.path.isdir(p)]
            snapshot_dir = snapshot_dirs[0] if snapshot_dirs else None

    if snapshot_dir is None:
        return None

    readme_path = os.path.join(snapshot_dir, "README.md")
    if not os.path.isfile(readme_path):
        return None

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return None

    return _parse_pipeline_tag_from_readme(content)


def _read_ref_commit(ref_path: str) -> Optional[str]:
    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_pipeline_tag_from_readme(content: str) -> Optional[str]:
    if not content.startswith("---"):
        return None
    lines = content.splitlines()
    meta_lines: List[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        meta_lines.append(line)
    for line in meta_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == "pipeline_tag":
            return value.strip().strip('"').strip("'")
    return None
