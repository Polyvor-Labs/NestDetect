from __future__ import annotations

from pathlib import Path

from nestdetect.dataset import (
    DatasetSpec,
    build_incremental_dataset,
    load_samples,
    materialize_dataset_with_id_mapping,
    validate_dataset,
)


def test_validate_valid_dataset(tmp_path: Path, make_dataset) -> None:
    data = make_dataset(
        tmp_path / "base",
        ["person", "chair"],
        [[(0, 0.5, 0.5, 0.25, 0.25)], [(1, 0.5, 0.5, 0.25, 0.25)]],
        [[(0, 0.5, 0.5, 0.25, 0.25), (1, 0.25, 0.25, 0.2, 0.2)]],
    )
    report = validate_dataset(data)
    assert report.valid
    assert report.instance_counts == {"person": 2, "chair": 2}
    assert report.image_counts == {"train": 2, "val": 1}


def test_validate_rejects_out_of_bounds_box(tmp_path: Path, make_dataset) -> None:
    data = make_dataset(
        tmp_path / "invalid",
        ["person"],
        [[(0, 0.95, 0.5, 0.2, 0.2)]],
        [[(0, 0.5, 0.5, 0.2, 0.2)]],
    )
    report = validate_dataset(data)
    assert not report.valid
    assert any(issue.code == "invalid_label" for issue in report.issues)


def test_validate_allows_small_rounding_tolerance(tmp_path: Path, make_dataset) -> None:
    data = make_dataset(
        tmp_path / "rounded",
        ["table"],
        [[(0, 0.694641, 0.859106, 0.610719, 0.281788)]],
        [[(0, 0.5, 0.5, 0.2, 0.2)]],
    )
    assert validate_dataset(data).valid


def test_incremental_builder_remaps_local_new_ids(tmp_path: Path, make_dataset) -> None:
    old = make_dataset(
        tmp_path / "old",
        ["person", "chair"],
        [[(0, 0.5, 0.5, 0.2, 0.2)], [(1, 0.5, 0.5, 0.2, 0.2)]],
        [[(0, 0.5, 0.5, 0.2, 0.2)]],
    )
    new = make_dataset(
        tmp_path / "new",
        ["laptop", "book"],
        [[(0, 0.5, 0.5, 0.2, 0.2)], [(1, 0.5, 0.5, 0.2, 0.2)]],
        [[(1, 0.5, 0.5, 0.2, 0.2)]],
    )
    output_yaml = build_incremental_dataset(
        new,
        tmp_path / "mix",
        replay_dataset_yaml=old,
        target_names=["person", "chair", "laptop", "book"],
    )
    spec = DatasetSpec.from_yaml(output_yaml)
    assert spec.names == ["person", "chair", "laptop", "book"]
    ids = {label.class_id for sample in load_samples(spec, "train") for label in sample.labels}
    assert ids == {0, 1, 2, 3}


def test_materialize_dataset_with_sparse_target_ids(tmp_path: Path, make_dataset) -> None:
    source = make_dataset(
        tmp_path / "source",
        ["person", "chair"],
        [[(0, 0.5, 0.5, 0.2, 0.2)], [(1, 0.5, 0.5, 0.2, 0.2)]],
        [[(1, 0.5, 0.5, 0.2, 0.2)]],
    )
    target_names = [f"class_{index}" for index in range(80)]
    target_names[0] = "person"
    target_names[56] = "chair"
    output_yaml = materialize_dataset_with_id_mapping(
        source,
        tmp_path / "coco80",
        target_names,
        {0: 0, 1: 56},
    )
    spec = DatasetSpec.from_yaml(output_yaml)
    assert len(spec.names) == 80
    ids = {label.class_id for sample in load_samples(spec, "train") for label in sample.labels}
    assert ids == {0, 56}
