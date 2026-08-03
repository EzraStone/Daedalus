"""Loading the gate library.

The geometry lives in ``gates.yaml`` rather than in code so that adding a gate
shape — or a whole second orientation set for a v2 layer — is a data change
that the placer picks up without knowing anything new.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import vocab as V

_LIBRARY_PATH = Path(__file__).with_name("gates.yaml")

_ATTACH_BY_NAME = {
    "floor": V.Attach.FLOOR,
    "north": V.Attach.NORTH,
    "south": V.Attach.SOUTH,
    "west": V.Attach.WEST,
    "east": V.Attach.EAST,
}
_DIR_BY_NAME = {
    "north": V.Dir4.NORTH,
    "south": V.Dir4.SOUTH,
    "west": V.Dir4.WEST,
    "east": V.Dir4.EAST,
}


@dataclass(frozen=True, slots=True)
class Orientation:
    """One placement variant of a gate.

    Randomising over orientations is the cheapest and most effective piece of
    augmentation available: a placer that only ever builds gates facing east
    teaches the model its own habits rather than the physics.
    """

    name: str
    block: tuple[int, int]
    torch: tuple[int, int]
    attach: V.Attach
    input_faces: tuple[tuple[int, int], ...]
    output_faces: tuple[tuple[int, int], ...]

    def cells(self) -> tuple[tuple[int, int], ...]:
        """The cells the gate itself occupies."""
        return (self.block, self.torch)


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    kind: str
    blocks: int
    latency_rt: int
    orientations: tuple[Orientation, ...]


@dataclass(frozen=True, slots=True)
class PortSpec:
    lever_x: int
    support_x: int
    lever_attach: V.Dir4
    repeater_x: int
    lamp_x: int
    output_facing: V.Dir4


@dataclass(frozen=True, slots=True)
class Library:
    version: int
    gates: dict[str, Gate]
    ports: PortSpec
    #: Longest dust run that still arrives at strength >= 1.
    max_dust_run: int
    repeater_latency_rt: int

    @property
    def inverter(self) -> Gate:
        return self.gates["invert"]


def _tuple2(pair) -> tuple[int, int]:
    x, z = pair
    return int(x), int(z)


@functools.lru_cache(maxsize=1)
def load(path: str | Path | None = None) -> Library:
    """Read and validate the gate library. Cached; the file is small and read
    once per process."""
    p = Path(path) if path else _LIBRARY_PATH
    raw = yaml.safe_load(p.read_text())

    gates: dict[str, Gate] = {}
    for g in raw["gates"]:
        orients = []
        for o in g["orientations"]:
            orients.append(
                Orientation(
                    name=o["name"],
                    block=_tuple2(o["block"]),
                    torch=_tuple2(o["torch"]),
                    attach=_ATTACH_BY_NAME[o["attach"]],
                    input_faces=tuple(_tuple2(f) for f in o["input_faces"]),
                    output_faces=tuple(_tuple2(f) for f in o["output_faces"]),
                )
            )
        cost = g.get("cost", {})
        gates[g["name"]] = Gate(
            name=g["name"],
            kind=g["kind"],
            blocks=int(cost.get("blocks", len(orients[0].cells()))),
            latency_rt=int(cost.get("latency_rt", 1)),
            orientations=tuple(orients),
        )

    passives = {p["name"]: p for p in raw.get("passives", [])}
    ports_raw = raw["ports"]
    ports = PortSpec(
        lever_x=int(ports_raw["input"]["lever_x"]),
        support_x=int(ports_raw["input"]["support_x"]),
        lever_attach=_DIR_BY_NAME[ports_raw["input"]["attach"]],
        repeater_x=int(ports_raw["output"]["repeater_x"]),
        lamp_x=int(ports_raw["output"]["lamp_x"]),
        output_facing=_DIR_BY_NAME[ports_raw["output"]["facing"]],
    )

    lib = Library(
        version=int(raw["version"]),
        gates=gates,
        ports=ports,
        max_dust_run=int(passives.get("dust", {}).get("max_run", 15)),
        repeater_latency_rt=int(passives.get("repeater", {}).get("cost", {}).get("latency_rt", 1)),
    )
    _validate(lib)
    return lib


def _validate(lib: Library) -> None:
    """Catch a library that describes a shape the simulator would reject."""
    if "invert" not in lib.gates:
        raise ValueError("the gate library must define an 'invert' cell")
    for gate in lib.gates.values():
        for o in gate.orientations:
            bx, bz = o.block
            tx, tz = o.torch
            step = (tx - bx, tz - bz)
            if abs(step[0]) + abs(step[1]) != 1:
                raise ValueError(f"{gate.name}/{o.name}: torch is not adjacent to its block")
            back = o.attach.delta
            if (back[0], back[2]) != (-step[0], -step[1]):
                raise ValueError(
                    f"{gate.name}/{o.name}: attach {o.attach.name} does not point back at the block"
                )
            occupied = set(o.cells())
            for face in o.input_faces:
                if face in occupied:
                    raise ValueError(f"{gate.name}/{o.name}: input face {face} is inside the gate")
                if abs(face[0] - bx) + abs(face[1] - bz) != 1:
                    raise ValueError(
                        f"{gate.name}/{o.name}: input face {face} is not adjacent to the block"
                    )
            for face in o.output_faces:
                if face in occupied:
                    raise ValueError(f"{gate.name}/{o.name}: output face {face} is inside the gate")
                if abs(face[0] - tx) + abs(face[1] - tz) != 1:
                    raise ValueError(
                        f"{gate.name}/{o.name}: output face {face} is not adjacent to the torch"
                    )
            # Faces of one torch must be mutually non-adjacent, or two nets on
            # different faces would merge and the fanout budget would be a lie.
            for i, a in enumerate(o.output_faces):
                for b in o.output_faces[i + 1 :]:
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                        raise ValueError(
                            f"{gate.name}/{o.name}: output faces {a} and {b} are adjacent"
                        )
