.PHONY: build release test test-fast test-full test-integration test-models check clean fmt lint demos bench bench-wire bench-routing bench-grid bench-all bench-save bench-check coverage coverage-summary larql-models-ci larql-models-test larql-models-fmt-check larql-models-lint larql-models-coverage-summary larql-models-bench-test

# Build
build:
	cargo build --workspace

release:
	cargo build --release -p larql-cli

# Test
#
# Default test target is intentionally fast: no integration binaries, no
# model-backed ignored tests. Use `test-full` for the historical full
# workspace run, and `test-models` for real-model/vindex checks.
test: test-fast

test-fast:
	cargo test --workspace --lib --bins

test-full:
	cargo test --workspace

test-integration:
	cargo test --workspace --tests

test-models:
	cargo test -p larql-inference --test test_arch_golden -- --ignored
	cargo test -p larql-inference --test test_logits_goldens -- --ignored
	cargo test -p larql-inference --test test_gemma3_smoke -- --ignored
	cargo test -p larql-inference --test test_generate_q4k_cpu -- --ignored
	cargo test -p larql-inference --test bench_probe_latency -- --ignored --nocapture
	cargo test -p larql-inference --test test_llm_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_constrained_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_trie_dispatch -- --ignored --nocapture

larql-models-test:
	cargo test -p larql-models

larql-models-fmt-check:
	cargo fmt -p larql-models -- --check

larql-models-lint:
	cargo clippy -p larql-models --all-targets -- -D warnings

larql-models-bench-test:
	cargo test -p larql-models --benches

larql-models-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-models --summary-only

larql-models-ci: larql-models-fmt-check larql-models-lint larql-models-test larql-models-bench-test

# Check (compile without building)
check:
	cargo check --workspace

# Code quality
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --tests -- -D warnings

# All quality checks
ci: fmt-check lint test-full

# Clean
clean:
	cargo clean

# Benchmarks
#
# `bench` runs the full quant_matvec suite and writes HTML reports under
# `target/criterion/`. `bench-save` records a baseline named `main`;
# `bench-check` re-runs and fails if any cell regresses past Criterion's
# default noise threshold. Plug `bench-check` into CI to catch the next
# 4× throughput cliff (the kind the q4_matvec_v4 row-drop bug caused) at
# PR time, not at goldens-fail time weeks later.
bench:
	cargo bench -p larql-compute --bench quant_matvec --features metal

bench-wire:
	cargo bench -p larql-inference --bench wire_codec

bench-routing:
	cargo bench -p larql-router --bench routing

bench-grid:
	./scripts/bench-grid-regress.sh $(MODEL)

bench-all: bench bench-wire bench-routing

bench-save:
	bash scripts/bench-regress.sh save

bench-check:
	bash scripts/bench-regress.sh check

# Demos
demos:
	cargo run --release -p larql-models --example architecture_demo
	cargo run --release -p larql-core --example graph_demo
	cargo run --release -p larql-core --example edge_demo
	cargo run --release -p larql-core --example serialization_demo
	cargo run --release -p larql-core --example algorithm_demo

demos-inference:
	cargo run --release -p larql-inference --example inference_demo

# Benchmarks
bench: bench-core

bench-core:
	cargo run --release -p larql-core --example bench_graph

bench-inference:
	cargo run --release -p larql-inference --example bench_inference

# Vindex micro-benches — synthetic, fast, safe under load.
bench-vindex:
	cargo bench -p larql-vindex --bench vindex_ops

# Vindex production-dim scaling bench. Refuses if larql-server / router
# are alive (they distort 1-2 GB matmuls). Run alone, on a cool host;
# results feed PERFORMANCE.md.
bench-vindex-scaling:
	@if pgrep -fl 'larql-(server|router)' >/dev/null 2>&1; then \
		echo "Refusing bench-vindex-scaling: larql daemons running. Stop them first."; \
		pgrep -fl 'larql-(server|router)'; \
		exit 2; \
	fi
	cargo bench -p larql-vindex --bench vindex_scaling

bench-all: bench-core bench-inference bench-vindex

# Coverage — uses cargo-llvm-cov (install with `cargo install cargo-llvm-cov`).
# Writes an HTML report to coverage/ that can be opened in a browser.
# Scoped to larql-vindex by default since the audit owner cares about
# that crate; pass CRATE=… to scope elsewhere.
COVERAGE_CRATE ?= larql-vindex
coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package $(COVERAGE_CRATE) --html --output-dir coverage
	@echo "Report: coverage/html/index.html"

coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; \
		exit 1; \
	fi
	cargo llvm-cov --package $(COVERAGE_CRATE) --summary-only

# Python extension (managed via uv)
python-setup:
	cd crates/larql-python && uv sync --no-install-project --group dev

python-build: python-setup
	cd crates/larql-python && uv run --no-sync maturin develop --release

python-test: python-build
	cd crates/larql-python && uv run --no-sync pytest tests/ -v

python-check:
	cargo check -p larql-python

python-clean:
	rm -rf crates/larql-python/.venv crates/larql-python/uv.lock

# Extraction
extract-test:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		--layer 26 -o output/test-L26.larql.json \
		--stats output/test-L26-stats.json

extract-full:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		-o output/gemma-3-4b-knowledge.larql.json \
		--stats output/gemma-3-4b-stats.json

# Inference
predict:
	cargo run --release -p larql-cli -- predict google/gemma-3-4b-it \
		--prompt "The capital of France is" -k 10
