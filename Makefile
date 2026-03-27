.PHONY: build release test check clean fmt lint docs

# Build
build:
	cargo build -p larql-core -p larql-cli

release:
	cargo build --release -p larql-cli

# Test
test:
	cargo test -p larql-core -p larql-cli

check:
	cargo check -p larql-core -p larql-cli
	cargo check -p larql-python

# Code quality
fmt:
	cargo fmt --all

lint:
	cargo clippy -p larql-core -p larql-cli -- -D warnings

# Clean
clean:
	cargo clean

# Python extension (requires virtualenv)
python-build:
	cd crates/larql-python && maturin develop --release

python-check:
	cargo check -p larql-python

# Extraction examples
extract-test:
	cargo run --release -p larql-cli -- weight-walk google/gemma-3-4b-it \
		--layer 26 -o output/test-L26.larql.json \
		--stats output/test-L26-stats.json

extract-full:
	cargo run --release -p larql-cli -- weight-walk google/gemma-3-4b-it \
		-o output/gemma-3-4b-knowledge.larql.json \
		--stats output/gemma-3-4b-stats.json
