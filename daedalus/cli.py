"""Command line entry point.

``python -m daedalus <command>``. The commands are the ones a stranger needs in
the first ten minutes: compile a sentence into a circuit, export it, build a
corpus, run the baselines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from . import __version__
from .redsim import Verifier, VerifierError
from .spec import Spec, SpecSyntaxError


def _read_spec(text: str) -> Spec:
    if Path(text).is_file():
        text = Path(text).read_text()
    return Spec.parse(text)


def cmd_compile(args) -> int:
    spec = _read_spec(args.spec)
    from .synth import compile as compile_spec

    rng = random.Random(args.seed)
    with Verifier() as v:
        attempt = compile_spec(spec, v, rng, attempts=args.attempts)
    if not attempt.ok:
        print(f"could not build this spec: {attempt.stage}: {attempt.detail}", file=sys.stderr)
        if attempt.stage in ("routing", "placement"):
            print(
                "\nRouting is placement-sensitive. Wire crossings use conservative\n"
                "seven-cell bridges, so try more attempts or another seed; a dense\n"
                "layout may still exceed the 16x6x16 build volume.",
                file=sys.stderr,
            )
        return 1

    print(spec.source())
    print()
    print(spec.table())
    print()
    print(attempt.grid.render())
    print()
    print(attempt.verdict)

    if args.out:
        from .schematic import block_summary, write_litematic, write_schem

        out = Path(args.out)
        writer = write_litematic if out.suffix == ".litematic" else write_schem
        writer(attempt.grid, out)
        print(f"\nwrote {out}")
        print("materials:", ", ".join(f"{k} x{v}" for k, v in block_summary(attempt.grid).items()))
    return 0


def cmd_verify(args) -> int:
    from .grid import Grid

    spec = _read_spec(args.spec)
    tokens = json.loads(Path(args.grid).read_text())
    placed = spec.default_placement()
    with Verifier() as v:
        verdict = v.evaluate(Grid.from_tokens(tokens), placed)
    print(verdict)
    return 0 if verdict.is_pass() else 1


def cmd_corpus(args) -> int:
    from .data.corpus import build

    report = build(args.out, seed=args.seed, scale=args.scale)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_baselines(args) -> int:
    from .data import sample_unique
    from .eval import ProceduralCompiler, Retrieval, Unconditional, grade, summarise
    from .synth import compile as compile_spec

    rng = random.Random(args.seed)
    specs = sample_unique(rng, args.specs)
    with Verifier() as v:
        corpus = []
        for s in sample_unique(random.Random(args.seed + 1000), args.specs * 2):
            placed = s.default_placement()
            a = compile_spec(s, v, rng, attempts=10, fixed_placement=placed)
            if a.ok:
                corpus.append((s, a.grid.tokens(), placed.input_z, placed.output_z))

        methods = [
            ProceduralCompiler(v, rng, attempts=args.attempts),
            Retrieval(corpus),
            Unconditional(rng),
        ]
        table = {}
        for method in methods:
            results, candidates = [], []
            for s in specs:
                placed = s.default_placement()
                cands = method(s, placed, args.k)
                candidates.append(cands)
                results.append(grade(cands, placed, v))
            table[method.name] = summarise(results, candidates)
    print(json.dumps(table, indent=2))
    return 0


def _module_present(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def cmd_serve(args) -> int:
    """Open the local window onto the compiler."""
    try:
        import uvicorn
    except ImportError:
        print(
            "the web window needs FastAPI and uvicorn:\n  pip install 'daedalus[web]'",
            file=sys.stderr,
        )
        return 2

    # uvicorn ships no WebSocket implementation of its own; without one it
    # answers upgrade requests with a bare 404 and the page sits there saying
    # "not connected" with nothing in the server log to explain it. Check up
    # front, where the fix is one line away.
    if not any(_module_present(name) for name in ("websockets", "wsproto")):
        print(
            "uvicorn has no WebSocket library, so the live view cannot connect.\n"
            "  pip install websockets",
            file=sys.stderr,
        )
        return 2

    # Fail here rather than on the first compile: "you forgot to build the
    # verifier" is the most likely reason a fresh clone does not work, and it
    # should not surface as a mysterious error three clicks later.
    from .redsim import find_binary

    find_binary()

    print(f"Daedalus on http://{args.host}:{args.port}")
    uvicorn.run(
        "daedalus.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


def cmd_tui(args) -> int:
    """The same window, in the terminal."""
    del args
    try:
        from .tui import main as tui_main
    except ImportError:
        print(
            "the terminal window needs Textual:\n  pip install 'daedalus[tui]'",
            file=sys.stderr,
        )
        return 2

    from .redsim import find_binary

    find_binary()
    return tui_main()


def cmd_selftest(args) -> int:
    del args
    try:
        with Verifier() as v:
            spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
            from .synth import compile as compile_spec

            attempt = compile_spec(spec, v, random.Random(0), attempts=20)
    except VerifierError as e:
        print(f"verifier unavailable: {e}", file=sys.stderr)
        return 2
    if not attempt.ok:
        print("selftest FAILED: could not build a NAND gate", file=sys.stderr)
        return 1
    print(attempt.grid.render())
    print(attempt.verdict)
    print("selftest ok")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="daedalus", description="Text-conditioned generation of verified redstone circuits."
    )
    parser.add_argument("--version", action="version", version=f"daedalus {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compile", help="build a circuit from a spec, and verify it")
    p.add_argument("spec", help="spec source, or a path to a file containing one")
    p.add_argument("--out", help="write a .schem or .litematic here")
    p.add_argument("--attempts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("verify", help="check a saved grid against a spec")
    p.add_argument("spec")
    p.add_argument("grid", help="JSON file holding 1536 token ids")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("corpus", help="build a training corpus")
    p.add_argument("out")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("baselines", help="run the non-learned baselines")
    p.add_argument("--specs", type=int, default=25)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--attempts", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("serve", help="open the local web window")
    # Loopback by default: there is no auth here, and there should not be.
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true", help="reload on source changes")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("tui", help="open the terminal window")
    p.set_defaults(func=cmd_tui)

    p = sub.add_parser("selftest", help="prove the whole pipeline works")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SpecSyntaxError as e:
        print(f"bad spec: {e}", file=sys.stderr)
        return 2
    except VerifierError as e:
        print(f"{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
