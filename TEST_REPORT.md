# Refactored Sequence Pipeline Validation

## Current Packaging Check

```text
python -m compileall -q .        passed
pytest -q                        
```

The package now declares runtime dependencies in `pyproject.toml`:

```text
numpy
pandas
scikit-learn
torch
```

The optional development dependency is:

```text
pytest
```

## What Was Revalidated Locally

- every Python file compiles
- legacy wrapper modules remain present
- new helper modules are import-addressable by path
- package and module filenames now use the `seqcore` surface
- stale coverage artifacts from the previous archive were removed to avoid reporting outdated coverage

## Commands Used

```bash
python -m compileall -q .\sequence_lab_generic_package

$env:PYTHONPATH = (Resolve-Path '.\sequence_lab_generic_package').Path
python -m pytest -q '.\sequence_lab_generic_package\tests'
```
