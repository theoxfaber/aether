# syntax=docker/dockerfile:1
FROM rust:1.82-slim-bookworm AS chef

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cargo install cargo-chef --locked

WORKDIR /app

FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY benches/ benches/
COPY src/ src/
COPY tests/ tests/
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release --bin aether-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/aether-server /usr/local/bin/aether-server

ENV AETHER_HOST=0.0.0.0
ENV AETHER_PORT=8080
ENV AETHER_CPU_ONLY=true

EXPOSE 8080

ENTRYPOINT ["aether-server"]
