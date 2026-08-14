"""Cross-language fidelity harness cases and reporting helpers."""

from daedalus import vocab as V
from daedalus.grid import Grid
from daedalus.redsim import Verifier
from harness.compare import build_bridge_case, simulate


def test_bridge_fidelity_case_is_elevated_and_passes_redsim():
    case = build_bridge_case()
    grid = Grid.from_tokens(case.tokens)

    assert any(
        grid.get(x, y, z) == V.WIRE
        for y in range(2, V.SY)
        for z in range(V.SZ)
        for x in range(V.SX)
    )

    with Verifier() as verifier:
        simulate(case, verifier)
    assert case.settled
    assert case.simulated == [0, 1, 2, 3]
