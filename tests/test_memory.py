from __future__ import annotations

from pathlib import Path

from nestdetect.dataset import DatasetSpec, load_samples
from nestdetect.memory import build_replay_memory, select_balanced_samples


def test_balanced_memory_represents_each_class(tmp_path: Path, make_dataset) -> None:
    data = make_dataset(
        tmp_path / "base",
        ["person", "chair", "table"],
        [
            [(0, 0.5, 0.5, 0.2, 0.2)],
            [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.2, 0.2)],
            [(1, 0.5, 0.5, 0.2, 0.2)],
            [(2, 0.5, 0.5, 0.2, 0.2)],
        ],
        [
            [(0, 0.5, 0.5, 0.2, 0.2)],
            [(1, 0.5, 0.5, 0.2, 0.2)],
            [(2, 0.5, 0.5, 0.2, 0.2)],
        ],
    )
    memory_yaml = build_replay_memory(
        data, tmp_path / "memory", train_per_class=1, val_per_class=1, seed=7
    )
    spec = DatasetSpec.from_yaml(memory_yaml)
    represented = {
        label.class_id for sample in load_samples(spec, "train") for label in sample.labels
    }
    assert represented == {0, 1, 2}
    assert (tmp_path / "memory" / "memory_index.csv").is_file()


def test_selection_is_deterministic(tmp_path: Path, make_dataset) -> None:
    data = make_dataset(
        tmp_path / "base",
        ["a", "b"],
        [
            [(0, 0.5, 0.5, 0.2, 0.2)],
            [(0, 0.5, 0.5, 0.2, 0.2)],
            [(1, 0.5, 0.5, 0.2, 0.2)],
            [(1, 0.5, 0.5, 0.2, 0.2)],
        ],
        [[(0, 0.5, 0.5, 0.2, 0.2)], [(1, 0.5, 0.5, 0.2, 0.2)]],
    )
    spec = DatasetSpec.from_yaml(data)
    samples = load_samples(spec, "train")
    first = select_balanced_samples(samples, 2, 1, seed=99)
    second = select_balanced_samples(samples, 2, 1, seed=99)
    assert [sample.image_path for sample in first] == [sample.image_path for sample in second]
