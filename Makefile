# Everything depends on the verifier, so it builds first.
.PHONY: all verifier test test-rust test-python lint selftest web corpus baselines clean

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
	ruff check daedalus tests || true

selftest: verifier
	./target/release/redsim selftest
	python3 -m daedalus selftest

# The local window: type a spec, watch it get placed, routed and verified.
web: verifier
	python3 -m daedalus serve

corpus: verifier
	python3 -m daedalus corpus data/ --scale 0.1

baselines: verifier
	python3 -m daedalus baselines --specs 25

# The number that makes everything else believable. Needs a Fabric server;
# see harness/README.md.
agreement: verifier
	python3 harness/compare.py --cases 10000 --out agreement.json

clean:
	cargo clean
	find . -name __pycache__ -type d -exec rm -rf {} +
