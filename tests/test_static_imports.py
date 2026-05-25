from __future__ import annotations

import importlib
import pathlib
import py_compile


def iter_python_files() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parents[1]
    return sorted(
        p
        for p in root.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
    )


def module_name_from_path(path: pathlib.Path) -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def test_every_python_file_compiles() -> None:
    for path in iter_python_files():
        py_compile.compile(str(path), doraise=True)


def test_every_module_imports() -> None:
    for path in iter_python_files():
        mod_name = module_name_from_path(path)
        if mod_name:
            importlib.import_module(mod_name)


def test_legacy_top_level_exports_resolve() -> None:
    from dataset import SequenceCorpus
    from model import SequenceStack, SequenceUnit, FeatureAdapter, OffsetEncoding
    from training import build_query_table, build_legacy_query_table, eval_epoch, masked_error, projection_error, train_one_epoch
    from utils import _select_columns, compute_global_stats, set_seed, split_subjects

    assert SequenceCorpus.__name__ == "SequenceCorpus"
    assert SequenceUnit.__name__ == "SequenceUnit"
    assert SequenceStack.__name__ == "SequenceStack"
    assert FeatureAdapter.__name__ == "FeatureAdapter"
    assert OffsetEncoding.__name__ == "OffsetEncoding"
    assert callable(train_one_epoch)
    assert callable(eval_epoch)
    assert callable(masked_error)
    assert callable(projection_error)
    assert callable(build_query_table)
    assert callable(build_legacy_query_table)
    assert callable(_select_columns)
    assert callable(compute_global_stats)
    assert callable(set_seed)
    assert callable(split_subjects)
