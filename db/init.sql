-- MLOps pipeline schema for processed data

CREATE TABLE sales (
    id              SERIAL PRIMARY KEY,
    type            TEXT,
    date            DATE NOT NULL,
    invoice_number  TEXT,
    customer        TEXT NOT NULL,
    item            TEXT NOT NULL,
    qty             INTEGER NOT NULL,
    unit_price      NUMERIC(12, 2),
    amount          NUMERIC(12, 2) NOT NULL,
    balance         NUMERIC(12, 2),
    sku             TEXT NOT NULL
);

CREATE INDEX idx_sales_date ON sales (date);
CREATE INDEX idx_sales_sku ON sales (sku);
CREATE INDEX idx_sales_sku_date ON sales (sku, date);

CREATE TABLE items (
    id              SERIAL PRIMARY KEY,
    item            TEXT NOT NULL,
    description     TEXT,
    type            TEXT,
    cost            NUMERIC(12, 2),
    price           NUMERIC(12, 2),
    tax_code        TEXT,
    qty_on_hand     INTEGER,
    qty_on_so       INTEGER,
    reorder_point   INTEGER,
    qty_on_po       INTEGER,
    uom             TEXT,
    preferred_vendor TEXT
);

CREATE TABLE customers (
    id              SERIAL PRIMARY KEY,
    customer        TEXT NOT NULL,
    bill_to         TEXT,
    primary_contact TEXT,
    main_phone      TEXT,
    fax             TEXT,
    balance_total   NUMERIC(12, 2)
);

CREATE TABLE features (
    id                      SERIAL PRIMARY KEY,
    date                    DATE NOT NULL,
    sku                     TEXT NOT NULL,
    qty                     INTEGER NOT NULL,
    amount                  NUMERIC(12, 2),
    order_count             INTEGER,
    qty_lag_1               NUMERIC(12, 2),
    qty_lag_2               NUMERIC(12, 2),
    qty_lag_3               NUMERIC(12, 2),
    qty_lag_4               NUMERIC(12, 2),
    qty_rolling_mean_4w     NUMERIC(12, 4),
    qty_rolling_std_4w      NUMERIC(12, 4),
    qty_rolling_mean_12w    NUMERIC(12, 4),
    qty_rolling_std_12w     NUMERIC(12, 4),
    week_of_year            INTEGER,
    month                   INTEGER,
    quarter                 INTEGER,
    month_sin               NUMERIC(8, 6),
    month_cos               NUMERIC(8, 6),
    week_sin                NUMERIC(8, 6),
    week_cos                NUMERIC(8, 6),
    sku_lifetime_mean       NUMERIC(12, 4),
    sku_lifetime_std        NUMERIC(12, 4),
    sku_lifetime_max        INTEGER,
    sku_active_rate         NUMERIC(8, 6),
    UNIQUE (date, sku)
);

CREATE INDEX idx_features_sku ON features (sku);
CREATE INDEX idx_features_date ON features (date);
