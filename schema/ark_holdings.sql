-- Apply to the development Replit PostgreSQL database.
-- Replit Publish copies development schema changes to the managed production DB.
CREATE TABLE IF NOT EXISTS ark_fund_snapshots (
    fund TEXT NOT NULL,
    report_date DATE NOT NULL,
    snapshot JSONB NOT NULL,
    saved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fund, report_date)
);

CREATE TABLE IF NOT EXISTS ark_fund_refresh (
    fund TEXT PRIMARY KEY,
    checked_at TIMESTAMPTZ NOT NULL,
    error TEXT,
    unchanged_on_refresh BOOLEAN NOT NULL DEFAULT FALSE
);