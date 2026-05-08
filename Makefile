.PHONY: build release test test-fast test-full test-integration test-models check clean fmt lint demos bench bench-core bench-inference bench-compute bench-wire bench-routing bench-grid bench-all bench-vindex bench-vindex-scaling bench-save bench-check coverage coverage-summary larql-core-ci larql-core-test larql-core-fmt-check larql-core-lint larql-core-feature-test larql-core-bench-test larql-core-bench larql-core-examples larql-core-coverage larql-core-coverage-html larql-models-ci larql-models-test larql-models-fmt-check larql-models-lint larql-models-coverage-summary larql-models-bench-test larql-vindex-ci larql-vindex-test larql-vindex-fmt-check larql-vindex-lint larql-vindex-examples larql-vindex-bench-test larql-vindex-bench larql-vindex-coverage larql-vindex-coverage-summary larql-vindex-coverage-html larql-vindex-coverage-policy larql-compute-test larql-compute-test-fast larql-compute-test-integration larql-compute-fmt-check larql-compute-lint larql-compute-ci larql-boundary-ci larql-boundary-test larql-boundary-fmt-check larql-boundary-lint larql-boundary-bench-test larql-boundary-examples

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

# larql-core — graph engine, algorithms, extraction helpers, serialization
larql-core-test:
	cargo test -p larql-core

larql-core-feature-test:
	cargo test -p larql-core --no-default-features
	cargo test -p larql-core --no-default-features --features msgpack

larql-core-fmt-check:
	cargo fmt -p larql-core -- --check

larql-core-lint:
	cargo clippy -p larql-core --all-targets -- -D warnings

larql-core-bench-test:
	cargo test -p larql-core --benches

larql-core-bench:
	cargo bench -p larql-core --bench graph

larql-core-examples:
	cargo run -p larql-core --example edge_demo
	cargo run -p larql-core --example graph_demo
	cargo run -p larql-core --example algorithm_demo
	cargo run -p larql-core --example filter_demo
	cargo run -p larql-core --example serialization_demo

larql-core-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-core --summary-only

larql-core-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-core --html --output-dir coverage/larql-core
	@echo "Report: coverage/larql-core/html/index.html"

larql-core-ci: larql-core-fmt-check larql-core-lint larql-core-test larql-core-feature-test larql-core-bench-test larql-core-examples

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

# larql-vindex - vindex extraction, storage, load/save, patch overlays
#
# Current local baseline: 68.18% line coverage from cargo-llvm-cov.
# Keep this as a ratchet: raise it when new coverage lands.
LARQL_VINDEX_COVERAGE_MIN ?= 68
LARQL_VINDEX_COVERAGE_POLICY ?= crates/larql-vindex/coverage-policy.json
LARQL_VINDEX_COVERAGE_REPORT ?= coverage/larql-vindex/summary.json

larql-vindex-test:
	cargo test -p larql-vindex

larql-vindex-fmt-check:
	cargo fmt -p larql-vindex -- --check

larql-vindex-lint:
	cargo clippy -p larql-vindex --all-targets -- -D warnings

larql-vindex-examples:
	cargo check -p larql-vindex --examples

larql-vindex-bench-test:
	cargo test -p larql-vindex --benches

larql-vindex-bench:
	cargo bench -p larql-vindex --bench vindex_ops

larql-vindex-coverage-policy:
	@if [ ! -f "$(LARQL_VINDEX_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_VINDEX_COVERAGE_REPORT)"; \
		echo "Run: make larql-vindex-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_VINDEX_COVERAGE_REPORT) $(LARQL_VINDEX_COVERAGE_POLICY)

larql-vindex-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	@mkdir -p coverage/larql-vindex
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy

larql-vindex-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --summary-only --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	@mkdir -p coverage/larql-vindex
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy

larql-vindex-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --html --output-dir coverage/larql-vindex --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy
	@echo "Report: coverage/larql-vindex/html/index.html"

larql-vindex-ci: larql-vindex-fmt-check larql-vindex-lint larql-vindex-test larql-vindex-examples larql-vindex-bench-test larql-vindex-coverage-summary

# larql-compute — CPU/Metal kernels and backend contracts
larql-compute-test: larql-compute-test-fast

larql-compute-test-fast:
	cargo test -p larql-compute --lib
	cargo test -p larql-compute --test test_backend_matmul_quant

larql-compute-test-integration:
	cargo test -p larql-compute --tests

larql-compute-fmt-check:
	cargo fmt -p larql-compute -- --check

larql-compute-lint:
	cargo clippy -p larql-compute --all-targets -- -D warnings

larql-compute-ci: larql-compute-fmt-check larql-compute-lint larql-compute-test-fast

# larql-boundary — confidence-gated BOUNDARY ref codec
larql-boundary-test:
	cargo test -p larql-boundary

larql-boundary-fmt-check:
	cargo fmt -p larql-boundary -- --check

larql-boundary-lint:
	cargo clippy -p larql-boundary --all-targets -- -D warnings

larql-boundary-bench-test:
	cargo test -p larql-boundary --benches

larql-boundary-examples:
	cargo run -p larql-boundary --example encode_decode
	cargo run -p larql-boundary --example gate_decision
	cargo run -p larql-boundary --example accuracy

larql-boundary-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-boundary --summary-only

larql-boundary-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-boundary --html --output-dir coverage/larql-boundary
	@echo "Report: coverage/larql-boundary/html/index.html"

larql-boundary-ci: larql-boundary-fmt-check larql-boundary-lint larql-boundary-test larql-boundary-bench-test larql-boundary-examples

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
# `bench` runs the core graph example. `bench-compute` runs the primary
# larql-compute Criterion surface. `bench-save` records a compute baseline
# named `main`; `bench-check` re-runs the compute benches and fails if any
# cell regresses past Criterion's default noise threshold.
bench: bench-core

bench-core:
	cargo run --release -p larql-core --example bench_graph

bench-inference:
	cargo run --release -p larql-inference --example bench_inference

# Compute kernel criterion bench (quant_matvec — Metal GPU).
bench-compute:
	cargo bench -p larql-compute --bench quant_matvec --features metal

# Wire codec criterion bench (encode/decode f32/f16/i8 throughput).
bench-wire:
	cargo bench -p larql-inference --bench wire_codec

# Router routing hot-path criterion bench (route/heartbeat/rebuild ns/op).
bench-routing:
	cargo bench -p larql-router --bench routing

# Grid end-to-end regression gate (requires LARQL_BENCH_FFN_URL env var).
bench-grid:
	./scripts/bench-grid-regress.sh $(MODEL)

bench-all: bench-core bench-inference bench-compute bench-wire bench-routing

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

bench-save:
	bash scripts/bench-regress.sh save

bench-check:
	bash scripts/bench-regress.sh check

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
