DATABASE_URL ?= postgres://raghub:raghub@localhost:55432/raghub?sslmode=disable
DATASET ?= datasets/smoke/v1.json
EVAL_OUTPUT ?= artifacts/evals/smoke.json

.PHONY: db-up db-down run test test-integration vet eval verify

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

eval:
	RAGHUB_DATABASE_URL="$(DATABASE_URL)" go run ./cmd/raghub-eval -migrate -dataset "$(DATASET)" -output "$(EVAL_OUTPUT)"

verify: vet test
