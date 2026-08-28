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
                "\nRouting failures are mostly structural rather than unlucky: on a\n"
                "random spec the solved fraction barely moves between 3 attempts and\n"
                "50. Another seed changes the port rows and sometimes helps; a bigger\n"
                "attempt count usually does not. See docs/benchmarks.md.",
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
        if out.suffix == ".json":
            _write_layout(out, spec, attempt.placed, attempt.grid.tokens())
        else:
            writer = write_litematic if out.suffix == ".litematic" else write_schem
            writer(attempt.grid, out)
        print(f"\nwrote {out}")
        print("materials:", ", ".join(f"{k} x{v}" for k, v in block_summary(attempt.grid).items()))
    return 0


def _write_layout(path: Path, spec: Spec, placed, tokens: list[int]) -> None:
    """Save a grid together with the port rows it was built against.

    The tokens alone are not enough to re-check a circuit. Port rows are a
    real degree of freedom and the compiler re-rolls them on every attempt, so
    they cannot be recovered from the seed -- which attempt won decides them.
    Writing them next to the grid is what makes `verify` exact instead of
    approximately right.
    """
    path.write_text(
        json.dumps(
            {
                "spec": spec.source(),
                "input_z": list(placed.input_z),
                "output_z": list(placed.output_z),
                "tokens": tokens,
            },
            indent=2,
        )
    )


def _load_layout(spec: Spec, path: Path):
    """Read a saved circuit and work out the placement it belongs to.

    Three formats reach this. A .json layout written by ``compile --out``
    carries its port rows and is exact. A schematic carries blocks only, so
    the rows are read off the faces. A bare list of ids carries neither, and
    says so.
    """
    from .grid import Grid

    if path.suffix in (".schem", ".litematic"):
        from .schematic import read_litematic, read_schem

        reader = read_litematic if path.suffix == ".litematic" else read_schem
        grid = reader(path)
        return grid, _placement_from_grid(spec, grid)

    blob = json.loads(path.read_text())
    if isinstance(blob, dict):
        return Grid.from_tokens(blob["tokens"]), spec.place(blob["input_z"], blob["output_z"])

    # A bare list of ids carries no port rows, so the best available guess is
    # the unjittered placement. That is frequently not the one the grid was
    # built against, and the result is a working circuit reported as a port
    # violation -- so say so rather than let it read as a real verdict.
    print(
        "note: this file has no port rows, so the default placement is assumed.\n"
        "      Re-export with `daedalus compile ... --out layout.json` for an\n"
        "      exact check.",
        file=sys.stderr,
    )
    return Grid.from_tokens(blob), spec.default_placement()


def cmd_verify(args) -> int:
    spec = _read_spec(args.spec)
    path = Path(args.grid)
    grid, placed = _load_layout(spec, path)
    if path.suffix in (".schem", ".litematic"):
        print(f"ports: inputs at rows {list(placed.input_z)}, "
              f"outputs at rows {list(placed.output_z)}")
    with Verifier() as v:
        verdict = v.evaluate(grid, placed)
    print(verdict)
    return 0 if verdict.is_pass() else 1


def cmd_power(args) -> int:
    """Show where the signal actually goes, one input assignment at a time."""
    from . import vocab as V
    from .render import power_layer

    spec = _read_spec(args.spec)
    grid, placed = _load_layout(spec, Path(args.layout))
    assignments = (
        [args.inputs] if args.inputs is not None else list(range(1 << spec.n_inputs))
    )

    with Verifier() as v:
        for mask in assignments:
            field = v.power(grid, placed, mask)
            bits = " ".join(
                f"{name}={mask >> k & 1}" for k, name in enumerate(spec.inputs)
            )
            lit = " ".join(
                f"{name}={field.outputs >> j & 1}" for j, name in enumerate(spec.outputs)
            )
            state = "settled" if field.settled else "UNSETTLED"
            print(f"\n{bits}  ->  {lit}   ({state} after {field.game_ticks} game ticks)")
            print(power_layer(grid.tokens(), field.dust, args.layer))
    print(
        f"\nStrengths decay one step per block, so a run reaching 0 within "
        f"{V.SZ} cells of its source wanted a repeater."
    )
    return 0


def _placement_from_grid(spec: Spec, grid):
    """Recover the port rows from a grid that came in without them.

    A schematic records blocks, not intent, so the only evidence of where the
    ports were meant to be is where the levers and lamps ended up. Reading
    them off the faces is exact when the circuit is one of ours and a clear
    error when it is not.
    """
    from . import vocab as V

    inputs = [z for z in range(V.SZ) if V.decode(grid.get(0, V.LOGIC_Y, z)).kind == "lever"]
    outputs = [
        z for z in range(V.SZ) if grid.get(V.OUTPUT_X, V.LOGIC_Y, z) == V.LAMP
    ]
    if len(inputs) != spec.n_inputs or len(outputs) != spec.n_outputs:
        raise SystemExit(
            f"this grid has {len(inputs)} lever(s) on the input face and "
            f"{len(outputs)} lamp(s) on the output face, but the spec declares "
            f"{spec.n_inputs} input(s) and {spec.n_outputs} output(s)"
        )
    return spec.place(inputs, outputs)


def cmd_corpus(args) -> int:
    from .data.corpus import build

    report = build(args.out, seed=args.seed, scale=args.scale)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_baselines(args) -> int:
    from .data import sample_unique
    from .eval import (
        ConstrainedRandom,
        ProceduralCompiler,
        Retrieval,
        Unconditional,
        grade,
        summarise,
    )
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
            ConstrainedRandom(rng),
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


def cmd_doctor(args) -> int:
    """Check everything a fresh clone needs, and say what is missing.

    The failure a newcomer actually hits is the verifier binary not being
    built, and they hit it three commands later as a stack trace. This asks
    every question up front and answers all of them, rather than stopping at
    the first thing that is wrong.
    """
    del args
    core_ok = True
    missing_extras: list[str] = []

    def report(name: str, good: bool, detail: str, required: bool = False) -> None:
        nonlocal core_ok
        if not good:
            if required:
                core_ok = False
            else:
                missing_extras.append(name)
        print(f"  {'ok  ' if good else 'MISS'}  {name:<22} {detail}")

    print("daedalus", __version__)

    try:
        from .redsim import find_binary

        binary = find_binary()
        report("verifier", True, str(binary), required=True)
    except VerifierError as e:
        report("verifier", False, f"{e}", required=True)
        binary = None

    if binary is not None:
        try:
            with Verifier() as v:
                spec = Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
                from .synth import compile as compile_spec

                attempt = compile_spec(spec, v, random.Random(0), attempts=30)
            report(
                "compiler",
                attempt.ok,
                str(attempt.verdict) if attempt.ok else f"could not build a NAND ({attempt.stage})",
                required=True,
            )
        except VerifierError as e:
            report("compiler", False, str(e), required=True)

    from .models import HAVE_TORCH

    if HAVE_TORCH:
        from .train import describe_device, pick_device

        device = pick_device("auto")
        report("torch", True, f"{device} — {describe_device(device)}")
    else:
        report("torch", False, "not installed — pip install 'daedalus[train]' (optional)")

    for extra, modules in (("web", ("fastapi", "uvicorn", "websockets")), ("tui", ("textual",))):
        missing = [m for m in modules if not _module_present(m)]
        report(
            extra,
            not missing,
            "ready" if not missing else f"missing {', '.join(missing)} (optional)",
        )

    # Only the verifier and the compiler are required. A missing extra is a
    # choice, not a broken install, so it must not fail the exit code.
    if not core_ok:
        print("\nnot ready — build the verifier with `cargo build --release -p redsim`")
        return 1
    if missing_extras:
        print(f"\nready. optional extras not installed: {', '.join(missing_extras)}")
    else:
        print("\nready")
    return 0


def cmd_bench(args) -> int:
    """Measure verifier throughput, so the README's number is reproducible."""
    import statistics
    import time

    from .synth import compile as compile_spec

    rng = random.Random(args.seed)
    spec = _read_spec(args.spec) if args.spec else Spec.parse("inputs A B\noutputs Q\nQ = !(A & B)")
    placed = spec.default_placement(rng)

    with Verifier() as v:
        attempt = compile_spec(spec, v, rng, attempts=30, fixed_placement=placed)
        if not attempt.ok:
            print("could not build a circuit to benchmark", file=sys.stderr)
            return 1
        grid = attempt.grid

        # Batched, because that is how the loop calls it: one request carrying
        # many grids amortises the pipe over the whole batch, and a per-call
        # figure measured one grid at a time is really a measure of the pipe.
        v.evaluate_batch([grid] * 8, placed)  # warm the worker

        per_call = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            v.evaluate_batch([grid] * args.batch, placed)
            per_call.append((time.perf_counter() - started) / args.batch)

    micros = sorted(x * 1e6 for x in per_call)
    print(f"grid: {attempt.verdict}")
    print(f"batch {args.batch} x {args.repeats} repeats\n")
    print(f"  median   {statistics.median(micros):8.1f} us/evaluation")
    print(f"  mean     {statistics.fmean(micros):8.1f} us")
    print(f"  fastest  {micros[0]:8.1f} us")
    print(f"  slowest  {micros[-1]:8.1f} us")
    print(f"\n  {1e6 / statistics.median(micros):,.0f} evaluations/second")
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
    from .train import TrainConfig, describe_device, evaluate, pick_device, train

    data = Path(args.corpus)
    train_set = load(data / "train.jsonl" if data.is_dir() else data)
    val_set = []
    if data.is_dir() and (data / "val.jsonl").exists():
        val_set = load(data / "val.jsonl")
    if not train_set:
        print(f"no examples in {data}", file=sys.stderr)
        return 1

    cls = AutoregressiveModel if args.model == "ar" else MaskedDiffusionModel
    if args.tiny:
        # Enough to prove the wiring on a laptop without an accelerator.
        cfg = ModelConfig(n_layers=2, d_model=128, n_heads=4, d_ff=256, nl_slots=args.nl_slots)
    else:
        cfg = ModelConfig(nl_slots=args.nl_slots)
    model = cls(cfg)

    device = pick_device(args.device)
    print(f"{args.model} · {model.body.n_parameters():,} parameters")
    print(f"device: {device} — {describe_device(device)}")
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
            prompt_dropout=args.prompt_dropout,
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
    if hasattr(model, "loss_at"):
        kwargs["guidance"] = args.guidance

    if args.prompt:
        if model.prompts is None:
            print(
                "this checkpoint was trained on the spec alone and has no prompt\n"
                "encoder. Retrain with `--nl-slots 4` to condition on text.",
                file=sys.stderr,
            )
            return 2
        from .text import encode_prompt

        features = torch.tensor(
            [encode_prompt(args.prompt, model.cfg.nl_length)] * args.k,
            dtype=torch.long,
            device=device,
        )
        kwargs["nl_embeddings"] = model.prompts(features)
        print(f"prompt: {args.prompt}")

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


def cmd_repair(args) -> int:
    """Break a working circuit, then ask the model to put it back.

    This is the operation masked diffusion exists for: the same forward pass
    as generation, with a different set of cells held fixed. An autoregressive
    model cannot do it at all, which is the comparison §07 is built around.
    """
    if not _need_torch():
        return 2
    import torch

    from . import tokens as T
    from . import vocab as V
    from .data.corpus import Example, corrupt
    from .grid import Grid
    from .synth import compile as compile_spec
    from .train import load_checkpoint

    spec = _read_spec(args.spec)
    rng = random.Random(args.seed)
    placed = spec.default_placement(rng)
    model = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    if not hasattr(model, "repair"):
        print("only the diffusion model can repair", file=sys.stderr)
        return 2

    with Verifier() as v:
        built = compile_spec(spec, v, rng, attempts=args.attempts, fixed_placement=placed)
        if not built.ok:
            print(f"could not build a circuit to damage: {built.stage}", file=sys.stderr)
            return 1
        working = built.grid.tokens()
        print(f"built    {built.verdict}")

        example = Example(
            spec_source=spec.source(),
            spec_hash=spec.key(),
            gates=spec.gates,
            n_inputs=spec.n_inputs,
            n_outputs=spec.n_outputs,
            rows=list(spec.rows),
            input_z=list(placed.input_z),
            output_z=list(placed.output_z),
            tokens=working,
            latency_rt=built.verdict.latency_rt,
            blocks=built.verdict.blocks,
            bbox=list(built.verdict.bbox),
            prompts=[],
        )
        damaged, hit = corrupt(example, rng, blocks=args.blocks)
        print(f"damaged  {v.evaluate(Grid.from_tokens(damaged), placed)}  ({len(hit)} cells)")

        torch.manual_seed(args.seed)
        prefix, _slots = T.spec_prefix(placed)
        device = next(model.parameters()).device
        out = model.repair(
            torch.tensor([prefix] * args.k, dtype=torch.long, device=device),
            torch.tensor([damaged], dtype=torch.long, device=device),
            hit,
            steps=args.steps,
            legality=T.legality_mask(placed),
            pinned=T.port_mask(placed),
        )
        candidates = [row.tolist() for row in out.cpu()]
        verdicts = v.evaluate_batch([Grid.from_tokens(c) for c in candidates], placed)

    print()
    for i, verdict in enumerate(verdicts):
        changed = sum(1 for a, b in zip(candidates[i], working) if a != b)
        print(f"  {i:>2}  {verdict}  ({changed} cells differ from the original)")
    fixed = [c for c, verdict in zip(candidates, verdicts) if verdict.is_pass()]
    print(f"\n{len(fixed)}/{args.k} repaired")
    if fixed:
        print(Grid.from_tokens(fixed[0]).render())
    del V
    return 0 if fixed else 1


def cmd_loop(args) -> int:
    """Run the verifier-guided self-improvement rounds of §06."""
    if not _need_torch():
        return 2
    from .train import LoopConfig, collapse_warning, load_checkpoint, run
    from .train.adapters import ModelSampler, ModelTrainer, anchors_from, spec_source

    model = load_checkpoint(args.checkpoint, device=args.device)

    anchors = []
    if args.corpus:
        from .data.corpus import load

        data = Path(args.corpus)
        anchors = anchors_from(load(data / "train.jsonl" if data.is_dir() else data))

    cfg = LoopConfig(
        rounds=args.rounds,
        specs_per_round=args.specs,
        candidates_per_spec=args.candidates,
        seed=args.seed,
    )
    with Verifier() as v:
        reports = run(
            ModelSampler(model, steps=args.steps),
            v,
            spec_source,
            cfg,
            trainer=ModelTrainer(model),
            anchors=anchors,
            out_dir=args.out,
        )

    print(f"{'round':>5} {'gates':>7} {'pass@1':>8} {'pass@k':>8} {'kept':>6} {'layouts/spec':>13}")
    for r in reports:
        gates = f"{r.difficulty[0]}-{r.difficulty[1]}"
        print(
            f"{r.round:>5} {gates:>7} {r.pass_at_1:>8.3f} {r.pass_at_k:>8.3f}"
            f" {r.accepted:>6} {r.layouts_per_spec:>13.2f}"
        )

    # Rising pass@1 with falling diversity is the failure mode that looks like
    # success, so it is reported rather than left for someone to notice.
    warning = collapse_warning(reports)
    if warning:
        print(f"\nWARNING: {warning}", file=sys.stderr)
    if args.out:
        print(f"\nwrote {Path(args.out) / 'rounds.jsonl'}")
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
    p.add_argument("--out", help="write a .schem, .litematic, or .json layout here")
    p.add_argument("--attempts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("verify", help="check a saved grid against a spec")
    p.add_argument("spec")
    p.add_argument("grid", help="a .json layout, or a .schem from anywhere")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("power", help="show where the signal goes")
    p.add_argument("spec", help="spec source, or a path to a file containing one")
    p.add_argument("layout", help="a .json layout or a .schem")
    p.add_argument("--inputs", type=int, help="one assignment as a bitmask (default: all)")
    p.add_argument("--layer", type=int, default=None, help="which y layer to draw")
    p.set_defaults(func=cmd_power)

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
    p.add_argument(
        "--nl-slots",
        type=int,
        default=0,
        help="condition on the corpus paraphrases as well as the spec (0 = spec only)",
    )
    p.add_argument(
        "--prompt-dropout",
        type=float,
        default=0.1,
        help="fraction of prompts blanked, so guidance has an unconditional branch",
    )
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("sample", help="generate from a checkpoint, and verify the result")
    p.add_argument("checkpoint", help="a model.pt written by `daedalus train`")
    p.add_argument("spec", help="spec source, or a path to a file containing one")
    p.add_argument("--out", help="write the first verified sample here")
    p.add_argument("-k", type=int, default=8, help="candidates to draw")
    p.add_argument("--steps", type=int, default=24, help="denoising steps (diffusion only)")
    p.add_argument("--prompt", help="condition on this text as well as the spec")
    p.add_argument(
        "--guidance",
        type=float,
        default=2.0,
        help="classifier-free guidance strength; 1.0 disables it",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("repair", help="damage a working circuit and have the model rebuild it")
    p.add_argument("checkpoint", help="a diffusion model.pt")
    p.add_argument("spec", help="spec source, or a path to a file containing one")
    p.add_argument("--blocks", type=int, default=6, help="cells to knock out")
    p.add_argument("-k", type=int, default=8, help="repair attempts to draw")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--attempts", type=int, default=30, help="compiler attempts to build one")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("doctor", help="check the install and say what is missing")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("bench", help="measure verifier throughput")
    p.add_argument("spec", nargs="?", help="spec to benchmark against (default: a NAND)")
    p.add_argument("--batch", type=int, default=64, help="grids per request")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("loop", help="run the verifier-guided self-improvement rounds")
    p.add_argument("checkpoint", help="a model.pt to start from")
    p.add_argument("--corpus", help="corpus to draw anchor examples from")
    p.add_argument("--out", help="write rounds.jsonl here")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--specs", type=int, default=200, help="specs per round")
    p.add_argument("--candidates", type=int, default=16, help="candidates per spec")
    p.add_argument("--steps", type=int, default=24, help="denoising steps (diffusion only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_loop)

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
