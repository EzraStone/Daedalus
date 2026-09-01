# Everything depends on the verifier, so it builds first.
.PHONY: all verifier test test-rust test-python lint selftest web tui corpus baselines \
	bench-compiler harness-setup harness-server harness-smoke \
        bench train sample loop repair agreement clean

all: verifier

verifier:
	cargo build --release -p redsim

test: test-rust test-python

test-rust:
	cargo test
	cargo test --release --test golden -- --nocapture 2>&1 | grep 'per evaluation' || true

test-python: verifier
	python3 -m pytest tests/ -q

lint:
	cargo clippy --all-targets -- -D warnings
	cargo fmt --check
	# No `|| true`. A lint target that cannot fail is not a lint target, and
	# CI runs the real thing anyway -- so this only ever hid a local failure
	# until the push.
	ruff check .

selftest: verifier
	./target/release/redsim selftest
	python3 -m daedalus selftest

# The local window: type a spec, watch it get placed, routed and verified.
web: verifier
	python3 -m daedalus serve

# The same window, without a browser.
tui: verifier
	python3 -m daedalus tui

corpus: verifier
	python3 -m daedalus corpus data/ --scale 0.1

baselines: verifier
	python3 -m daedalus baselines --specs 25

# Verifier throughput. The figure the whole approach rests on, measured
# rather than asserted -- see docs/benchmarks.md.
bench: verifier
	python3 -m daedalus bench --batch 64

# The other half of the picture. A verdict is microseconds and a layout is
# a quarter of a second, so the target above says nothing about how long
# building a corpus takes.
bench-compiler: verifier
	python3 -m daedalus bench --compiler --specs 50

# Needs the training extra: pip install -e ".[train]"
train: verifier
	python3 -m daedalus train data/ --out runs/first

sample: verifier
	python3 -m daedalus sample runs/first/model.pt specs/nand.txt -k 8

# Damage a working circuit and have the model rebuild it. The operation
# masked diffusion exists for.
repair: verifier
	python3 -m daedalus repair runs/first/model.pt specs/nand.txt

# The section 06 rounds: sample, verify, keep what passes, retrain.
loop: verifier
	python3 -m daedalus loop runs/first/model.pt --corpus data/ --rounds 5

# The number that makes everything else believable. Needs a Fabric server;
# see harness/README.md.
# The three steps to the number the whole repository is waiting on. They need
# a network and about a gigabyte of downloads; nothing else in this Makefile
# does, which is why they are not wired into `all`.
harness-setup:
	harness/server/setup.sh --accept-eula

harness-server:
	harness/server/start.sh

harness-smoke:
	harness/server/smoke.sh --cases 10

agreement: verifier
	python3 harness/compare.py --cases 10000 --out agreement.json

clean:
	cargo clean
	find . -name __pycache__ -type d -exec rm -rf {} +
