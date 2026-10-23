"""``hf_dataset`` revision pinning: a dataset ``datasets`` loaded from another cached revision is
replaced by the pinned revision's Arrow files, or rejected when those are not cached."""

from __future__ import annotations

import pytest
from datasets import Dataset

from medpsy_eval.remote import _at_revision

FALLBACK, PINNED, MISSING = "a" * 40, "b" * 40, "c" * 40


@pytest.fixture
def fallback(tmp_path) -> Dataset:
    for rev, question in ((FALLBACK, "newer"), (PINNED, "pinned")):
        Dataset.from_list([{"q": question}]).save_to_disk(tmp_path / "default" / rev)
    return Dataset.from_file(str(tmp_path / "default" / FALLBACK / "data-00000-of-00001.arrow"))


def test_loaded_revision_is_kept(fallback) -> None:
    assert _at_revision(fallback, "org/ds", FALLBACK) is fallback


def test_other_cached_revision_is_replaced_by_the_pinned_one(fallback) -> None:
    assert _at_revision(fallback, "org/ds", PINNED)["q"] == ["pinned"]


def test_uncached_pinned_revision_is_an_error(fallback) -> None:
    with pytest.raises(RuntimeError, match=f"org/ds@{MISSING} is not in the local datasets cache"):
        _at_revision(fallback, "org/ds", MISSING)


def test_in_memory_dataset_is_kept() -> None:
    ds = Dataset.from_list([{"q": "x"}])
    assert _at_revision(ds, "org/ds", PINNED) is ds
