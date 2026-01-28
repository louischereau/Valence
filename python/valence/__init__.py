"""Valence: High-performance GNN engine."""

# Prefer the PyO3 extension named `valence`; fall back to `_lowlevel`.
try:  # Built via default pyo3 module name
    from .valence import MolecularGraph  # type: ignore
except Exception:
    try:  # Built via pyproject's `tool.maturin.module-name = "valence._lowlevel"`
        from ._lowlevel import MolecularGraph  # type: ignore
    except Exception as _e:  # noqa: F841
        # Extension module may not be built yet (e.g., source install only).
        MolecularGraph = None  # type: ignore

__all__ = ["MolecularGraph"]
