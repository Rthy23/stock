-- Apply to development only. Managed production schema changes go through Publish.
CREATE TABLE IF NOT EXISTS research_runs (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    background boolean NOT NULL,
    method text NOT NULL,
    payload jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS research_scores (
    run_id uuid NOT NULL REFERENCES research_runs(id),
    ticker text NOT NULL,
    observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    factors jsonb NOT NULL,
    PRIMARY KEY (run_id, ticker)
);
CREATE INDEX IF NOT EXISTS research_scores_observed ON research_scores(observed_at);
CREATE INDEX IF NOT EXISTS research_scores_ticker_time ON research_scores(ticker,observed_at);
CREATE TABLE IF NOT EXISTS research_prices (
    ticker text NOT NULL,
    session_close timestamptz NOT NULL,
    adjusted_close double precision NOT NULL CHECK (adjusted_close > 0),
    observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (ticker, session_close)
);
CREATE TABLE IF NOT EXISTS research_alerts (
    event_key text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    ticker text,
    kind text NOT NULL,
    message text NOT NULL
);
CREATE TABLE IF NOT EXISTS research_settings (
    key text PRIMARY KEY,
    value jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS research_worker (
    id integer PRIMARY KEY CHECK (id=1),
    started_at timestamptz,
    completed_at timestamptz,
    last_error text,
    coverage double precision
);
