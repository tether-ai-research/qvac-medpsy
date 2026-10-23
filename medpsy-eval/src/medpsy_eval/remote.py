"""Pinned remote data: Hugging Face datasets at a fixed revision."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def hf_dataset(path: str, *args: Any, revision: str, **kwargs: Any) -> Any:
    """``load_dataset(path, *args, revision=revision, **kwargs)``, guaranteed to be ``revision``.

    When the Hub is unreachable (``HF_HUB_OFFLINE=1`` or a network error), ``datasets`` ignores
    ``revision`` and loads the most recently cached revision. The cache keeps each revision's
    Arrow files in a directory named by its commit hash, so a fallback is detected from the
    loaded files and replaced by the pinned revision's files, or fails if those are not cached.
    """
    return _at_revision(load_dataset(path, *args, revision=revision, **kwargs), path, revision)


_SHARD = re.compile(r"-\d{5}-of-\d{5}$")


def _at_revision(ds: Any, path: str, revision: str) -> Any:
    if isinstance(ds, DatasetDict):
        return DatasetDict({k: _at_revision(v, path, revision) for k, v in ds.items()})
    files = [Path(f["filename"]) for f in ds.cache_files]
    if not files or files[0].parent.name == revision:
        return ds
    pinned = files[0].parent.with_name(revision)
    stems = sorted({_SHARD.sub("", f.stem) for f in files})
    shards = sorted(p for p in pinned.glob("*.arrow") if _SHARD.sub("", p.stem) in stems)
    if not shards:
        raise RuntimeError(
            f"{path}@{revision} is not in the local datasets cache ({pinned}); datasets loaded "
            f"revision {files[0].parent.name} instead. Load it once with Hub access.")
    return concatenate_datasets([Dataset.from_file(str(p), split=ds.split) for p in shards])
