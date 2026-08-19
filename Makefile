DATABASE_URL ?= postgres://raghub:raghub@localhost:55432/raghub?sslmode=disable
DATASET ?= datasets/smoke/v3.json
FTS_EVAL_OUTPUT ?= artifacts/evals/v3-fts.json
DENSE_EVAL_OUTPUT ?= artifacts/evals/v3-dense.json
HYBRID_EVAL_OUTPUT ?= artifacts/evals/v3-hybrid.json
COMPARE_EVAL_OUTPUT ?= artifacts/evals/v3-three-way.json
EVAL_BINARY ?= /tmp/raghub-eval

.PHONY: db-up db-down run test test-integration vet eval eval-build eval-fts eval-dense eval-hybrid eval-compare eval-clean-revision eval-three-way eval-paired eval-v2-regression eval-all verify

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

run:
	RAGHUB_DATABASE_URL="$(DATABASE_URL)" go run ./cmd/raghub-api -migrate

test:
	go test ./...

test-integration:
	RAGHUB_TEST_DATABASE_URL="$(DATABASE_URL)" go test -count=1 ./internal/store/postgres

vet:
	go vet ./...

eval: eval-fts

eval-build:
	go build -o "$(EVAL_BINARY)" ./cmd/raghub-eval

eval-fts: eval-build
	RAGHUB_DATABASE_URL="$(DATABASE_URL)" "$(EVAL_BINARY)" -migrate -mode fts -dataset "$(DATASET)" -output "$(FTS_EVAL_OUTPUT)"

eval-dense: eval-build
	RAGHUB_DATABASE_URL="$(DATABASE_URL)" "$(EVAL_BINARY)" -migrate -mode dense -dataset "$(DATASET)" -output "$(DENSE_EVAL_OUTPUT)"

eval-hybrid: eval-build
	RAGHUB_DATABASE_URL="$(DATABASE_URL)" "$(EVAL_BINARY)" -migrate -mode hybrid -dataset "$(DATASET)" -output "$(HYBRID_EVAL_OUTPUT)"

eval-compare:
	go run ./cmd/raghub-eval-compare -fts "$(FTS_EVAL_OUTPUT)" -dense "$(DENSE_EVAL_OUTPUT)" -hybrid "$(HYBRID_EVAL_OUTPUT)" -output "$(COMPARE_EVAL_OUTPUT)"

eval-clean-revision:
	@test -n "$$(git rev-parse --verify HEAD 2>/dev/null)" || (echo "three-way evaluation requires a committed Git revision"; exit 1)
	@test -z "$$(git status --porcelain)" || (echo "three-way evaluation requires a clean Git worktree"; exit 1)

eval-three-way: eval-clean-revision
	$(MAKE) eval-fts
	$(MAKE) eval-dense
	$(MAKE) eval-hybrid
	$(MAKE) eval-compare

# Backward-compatible target name; the accepted protocol now has three sides.
eval-paired: eval-three-way

eval-v2-regression:
	$(MAKE) eval-three-way \
		DATASET=datasets/smoke/v2.json \
		FTS_EVAL_OUTPUT=artifacts/evals/v2-fts.json \
		DENSE_EVAL_OUTPUT=artifacts/evals/v2-dense.json \
		HYBRID_EVAL_OUTPUT=artifacts/evals/v2-hybrid.json \
		COMPARE_EVAL_OUTPUT=artifacts/evals/v2-three-way.json

eval-all:
	$(MAKE) eval-three-way
	$(MAKE) eval-v2-regression

verify: vet test
