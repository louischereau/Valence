import os
import sys

# Ensure source package is importable without building the wheel
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

import pytest
import valence


def test_package_import():
    # The symbol should exist in the package namespace
    assert hasattr(valence, "MolecularGraph")


def test_molecular_graph_smoke():
    mg = getattr(valence, "MolecularGraph", None)
    if mg is None:
        pytest.skip("MolecularGraph extension not built; skipping low-level test")

    # Two hydrogen atoms ~0.74 Å apart should form a bond under 1.0 Å cutoff
    g = mg([1, 1], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
    bonds = g.find_bonds(1.0)
    assert len(bonds) == 1
    i, j, d = bonds[0]
    assert {i, j} == {0, 1}
    assert 0.7 < d < 0.8


def test_molecular_graph_no_bond_when_far():
    mg = getattr(valence, "MolecularGraph", None)
    if mg is None:
        pytest.skip("MolecularGraph extension not built; skipping low-level test")

    g = mg([1, 1], [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    bonds = g.find_bonds(1.0)
    assert bonds == []
