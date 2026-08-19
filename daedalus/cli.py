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


def _need_torch() -> bool:
    from .models import HAVE_TORCH

    if not HAVE_TORCH:
        print(
            "this needs PyTorch:\n  pip install 'daedalus[train]'\n"
            "For the RX 7600 see docs/hardware.md.",
            file=sys.stderr,
        )
    return HAVE_TORCH


def cmd_train(args) -> int:
    """Pretrain a generator on a procedural corpus."""
    if not _need_torch():
        return 2
    from .data.corpus import load
    from .models import AutoregressiveModel, MaskedDiffusionModel, ModelConfig
    from .train import TrainConfig, evaluate, train

    data = Path(args.corpus)
    train_set = load(data / "train.jsonl" if data.is_dir() else data)
    val_set = []
    if data.is_dir() and (data / "val.jsonl").exists():
        val_set = load(data / "val.jsonl")
    if not train_set:
        print(f"no examples in {data}", file=sys.stderr)
        return 1

    cls = AutoregressiveModel if args.model == "ar" else MaskedDiffusionModel
    cfg = ModelConfig()
    if args.tiny:
        # Enough to prove the wiring on a laptop without an accelerator.
        cfg = ModelConfig(n_layers=2, d_model=128, n_heads=4, d_ff=256)
    model = cls(cfg)

    print(f"{args.model} · {model.body.n_parameters():,} parameters")
    print(f"{len(train_set)} training examples, {len(val_set)} validation")
    history = train(
        model,
        train_set,
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
        ),
        out_dir=args.out,
        val=val_set or None,
    )
    for entry in history:
        line = f"  step {entry['step']:>5}  loss {entry['loss']:.4f}"
        if "val_loss" in entry:
            line += f"  val {entry['val_loss']:.4f}"
        print(line)
    if val_set:
        # The comparable number. Training loss is not one for the diffusion
        # model -- see daedalus.train.evaluate.
        print(f"\nfinal validation loss: {evaluate(model, val_set, seed=args.seed):.4f}")
    if args.out:
        print(f"wrote {Path(args.out) / 'model.pt'}")
    return 0


def cmd_sample(args) -> int:
    """Generate candidate grids from a checkpoint and verify every one."""
    if not _need_torch():
        return 2
    import torch

    from . import tokens as T
    from .eval import grade
    from .grid import Grid
    from .train import load_checkpoint

    spec = _read_spec(args.spec)
    placed = spec.default_placement(random.Random(args.seed))
    model = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    torch.manual_seed(args.seed)
    prefix, _slots = T.spec_prefix(placed)
    device = next(model.parameters()).device
    batch = torch.tensor([prefix] * args.k, dtype=torch.long, device=device)
    kwargs = {"steps": args.steps} if hasattr(model, "loss_at") else {}
    bodies = model.sample(
        batch,
        legality=T.legality_mask(placed),
        pinned=T.port_mask(placed),
        **kwargs,
    )
    candidates = [row.tolist() for row in bodies.cpu()]

    with Verifier() as v:
        result = grade(candidates, placed, v)
        verdicts = [v.evaluate(Grid.from_tokens(c), placed) for c in candidates]

    print(spec.source())
    print()
    for i, verdict in enumerate(verdicts):
        print(f"  {i:>2}  {verdict}")
    passed = [c for c, verdict in zip(candidates, verdicts) if verdict.is_pass()]
    print(f"\n{len(passed)}/{args.k} verified")
    if passed and args.out:
        from .schematic import write_litematic, write_schem

        out = Path(args.out)
        writer = write_litematic if out.suffix == ".litematic" else write_schem
        writer(Grid.from_tokens(passed[0]), out)
        print(f"wrote {out}")
    if passed:
        print(Grid.from_tokens(passed[0]).render())
    del result
    return 0 if passed else 1


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

    p = sub.add_parser("train", help="pretrain a generator on a corpus")
    p.add_argument("corpus", help="corpus directory, or a single .jsonl file")
    p.add_argument("--model", choices=("mdm", "ar"), default="mdm")
    p.add_argument("--out", help="write model.pt and history.json here")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, ...")
    p.add_argument("--tiny", action="store_true", help="small config for a CPU wiring check")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("sample", help="generate from a checkpoint, and verify the result")
    p.add_argument("checkpoint", help="a model.pt written by `daedalus train`")
    p.add_argument("spec", help="spec source, or a path to a file containing one")
    p.add_argument("--out", help="write the first verified sample here")
    p.add_argument("-k", type=int, default=8, help="candidates to draw")
    p.add_argument("--steps", type=int, default=24, help="denoising steps (diffusion only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_sample)

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
