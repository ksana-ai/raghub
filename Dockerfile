# syntax=docker/dockerfile:1
FROM golang:1.27.1-alpine AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download && go mod verify
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /out/raghub-api ./cmd/raghub-api

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/raghub-api /usr/local/bin/raghub-api
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/raghub-api"]
