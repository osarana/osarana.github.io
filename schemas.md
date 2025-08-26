# CSV Schemas

This project uses three CSVs that are easy to ingest into a vector DB and to train the two models (ANN-A gate and ANN-B selector).

- `snapshots.csv` — one row per decision tick (global context).
- `candidates.csv` — one row per **snapshot × candidate network** (LTE, WiFi).
- `embeddings.csv` — precomputed RF+context vectors (one row per **snapshot × candidate**).

> Conventions  
> • Timestamps are ISO 8601 strings.  
> • Missing RF values for a technology (e.g., LTE’s RSSI, WiFi’s RSRP) are stored as empty string `""` to stay CSV-friendly.  
> • Numeric columns are `float` unless noted, IDs/enums are `string`, counters are `int`, booleans use `0/1`.  
> • `snapshot_id` joins tables.

---

## 1) `snapshots.csv` (context per decision tick)

| Column          | Type   | Description                                                                 |
|-----------------|--------|-----------------------------------------------------------------------------|
| `snapshot_id`   | int    | Monotonic tick ID starting at 0.                                            |
| `timestamp`     | string | ISO 8601 timestamp for this tick.                                           |
| `lat`           | float  | GPS latitude (coarse).                                                       |
| `lon`           | float  | GPS longitude (coarse).                                                      |
| `geo_bucket`    | string | Coarse geohash-like bucket, e.g. `"25.76_-80.19"` (~1 km cells).            |
| `speed_mps`     | float  | Device speed (meters/second).                                               |
| `heading_deg`   | float  | Heading (0–360 degrees).                                                     |
| `weekday`       | int    | Day of week (0=Mon … 6=Sun).                                                |
| `hour`          | int    | Hour of day (0–23).                                                          |
| `app_class`     | string | One of: `"voip"`, `"map"`, `"bg"`.                                          |
| `battery_pct`   | int    | Battery level (0–100).                                                       |
| `current_network` | string | The network currently connected: `"LTE"` or `"WiFi"` (chosen per tick).   |

**Primary key:** `(snapshot_id)`.

---

## 2) `candidates.csv` (per snapshot × candidate network)

One row per candidate network (LTE or WiFi) evaluated at `snapshot_id`.

| Column               | Type   | Description                                                                                           |
|----------------------|--------|-------------------------------------------------------------------------------------------------------|
| `snapshot_id`        | int    | Join key to `snapshots.csv`.                                                                          |
| `candidate_network`  | string | `"LTE"` or `"WiFi"`.                                                                                  |
| `available`          | int    | 1 if network is available/visible at this tick; else 0.                                               |
| `ambient_load`       | float  | 0..1 proxy for local congestion.                                                                      |
| **RF (cellular-only)** |||
| `rsrp_dbm`           | float\|"" | LTE RSRP (dBm). Empty `""` for WiFi rows.                                                           |
| `rsrq_db`            | float\|"" | LTE RSRQ (dB). Empty `""` for WiFi rows.                                                            |
| `sinr_db`            | float\|"" | LTE SINR (dB). Empty `""` for WiFi rows.                                                            |
| **RF (wifi-only)**    |||
| `rssi_dbm`           | float\|"" | WiFi RSSI (dBm). Empty `""` for LTE rows.                                                           |
| `snr_db`             | float\|"" | WiFi SNR (dB). Empty `""` for LTE rows.                                                             |
| **Common RF/meta**    |||
| `ber`                | float  | Bit error proxy (0..~0.2).                                                                            |
| `bler`               | float  | Block error proxy (0..~0.3).                                                                          |
| `band`               | int    | LTE band (e.g., 3,7,28,41,66) or WiFi GHz flag (2/5/6).                                               |
| `chan_bw_mhz`        | int    | Channel bandwidth (MHz).                                                                              |
| `access_denials_5m`  | int    | Count of access/auth/assoc denials over last 5 minutes.                                              |
| `last_denial_s`      | int    | Seconds since last denial.                                                                            |
| `p_denial`           | float  | Estimated probability of denial (0..0.35).                                                            |
| **Performance**       |||
| `tput_dl_mbps`       | float  | Predicted/realized DL throughput (Mbps).                                                              |
| `tput_ul_mbps`       | float  | Predicted/realized UL throughput (Mbps).                                                              |
| `rtt_ms`             | float  | Round-trip time (ms).                                                                                 |
| `jitter_ms`          | float  | Jitter (ms).                                                                                          |
| `loss_rate`          | float  | Packet loss (0..0.3).                                                                                 |
| **Cost/Energy**       |||
| `energy_mwh_per_mb`  | float  | Energy per MB (mWh/MB).                                                                               |
| `cost_per_mb`        | float  | Monetary cost per MB.                                                                                 |
| **Utility & labels**  |||
| `standalone_utility` | float  | Utility without switching penalty (higher is better).                                                 |
| `switch_outage_ms`   | float  | Expected outage if switching from `current_network` to `candidate_network` (0 if same).              |
| `net_utility`        | float  | Utility including `switch_outage_ms`.                                                                 |
| `label_best_network` | string | Hindsight-best network for this `snapshot_id` (argmax of `net_utility` among `available==1`).        |
| `label_best_net_utility` | float | The best `net_utility` value at this snapshot.                                                     |
| `label_switch_now`   | int    | 1 if switching to this candidate beats staying on current by a margin; else 0.                       |

**Primary key:** `(snapshot_id, candidate_network)`.

**Notes**
- For LTE rows, WiFi-only columns are `""`; for WiFi rows, LTE-only columns are `""`.  
- During training, cast `""` to `NaN`/0.0 as you prefer (the generator uses `""` to keep CSV tidy).

---

## 3) `embeddings.csv` (vector rows for kNN / vector DB)

Precomputed numeric embeddings per `(snapshot_id, candidate_network)` to support RAG/kNN.  
These are normalized RF+context features with a tanh expansion.

| Column              | Type   | Description                                          |
|---------------------|--------|------------------------------------------------------|
| `snapshot_id`       | int    | Join key.                                           |
| `candidate_network` | string | `"LTE"` or `"WiFi"`.                                |
| `e0..eK`            | float  | Embedding dimensions (size depends on build logic). |

**Primary key:** `(snapshot_id, candidate_network)`.

**Construction (reference)**
- Numeric features included: `rsrp_dbm, rsrq_db, sinr_db, rssi_dbm, snr_db, hour, weekday, speed_mps`.  
- Each column is z-scored; tanh of each is concatenated → `2×features` dims.

---

## Minimal headers (copy-paste)

**snapshots.csv**
