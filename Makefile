DATABASE_URL ?= postgres://raghub:raghub@localhost:55432/raghub?sslmode=disable
DATASET ?= datasets/smoke/v2.json
FTS_EVAL_OUTPUT ?= artifacts/evals/v2-fts.json
DENSE_EVAL_OUTPUT ?= artifacts/evals/v2-dense.json
COMPARE_EVAL_OUTPUT ?= artifacts/evals/v2-fts-vs-dense.json

.PHONY: db-up db-down run test test-integration vet eval eval-fts eval-dense eval-compare eval-clean-revision eval-paired verify

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

eval-fts:
	RAGHUB_DATABASE_URL="$(DATABASE_URL)" go run ./cmd/raghub-eval -migrate -mode fts -dataset "$(DATASET)" -output "$(FTS_EVAL_OUTPUT)"

eval-dense:
	RAGHUB_DATABASE_URL="$(DATABASE_URL)" go run ./cmd/raghub-eval -migrate -mode dense -dataset "$(DATASET)" -output "$(DENSE_EVAL_OUTPUT)"

eval-compare:
	go run ./cmd/raghub-eval-compare -baseline "$(FTS_EVAL_OUTPUT)" -candidate "$(DENSE_EVAL_OUTPUT)" -output "$(COMPARE_EVAL_OUTPUT)"

eval-clean-revision:
	@test -n "$$(git rev-parse --verify HEAD 2>/dev/null)" || (echo "eval-paired requires a committed Git revision"; exit 1)
	@test -z "$$(git status --porcelain)" || (echo "eval-paired requires a clean Git worktree"; exit 1)

eval-paired: eval-clean-revision
	$(MAKE) eval-fts
	$(MAKE) eval-dense
	$(MAKE) eval-compare

verify: vet test
