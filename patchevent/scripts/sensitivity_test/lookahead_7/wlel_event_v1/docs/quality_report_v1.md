# WLEL Data Quality Report (v1)

- Generated at: 2026-04-03T20:41:27
- Time range: 2021-01-01 00:00:00 -> 2025-11-27 16:00:00
- Total rows: 43001

## Data Cleaning
- Duplicate timestamps removed: 0
- Missing/inserted points filled: 10
- Irregular interval ratio (after unification): 0.000000

## Data Quality
- Missing rate (value): 0.000000
- Outlier rate (robust z): 0.004698
- Legacy peak density (`is_peak`): 0.048627
- Event window density (`is_peak_event`): 0.232460

## Event Summary
- Event count: 2091
- Event density per 1k points: 48.6268
- Duration mean/p50/p90/max: 4.780 / 5.000 / 5.000 / 5
- Apex mean/p90/max: 6050.542 / 8200.146 / 10541.681

## Threshold Config (locked for v1)
- event_mode: gradient_width
- boundary_semantics: threshold_crossing_hysteresis
- merge_policy: gap_based
- merge_gap_effective: 1
- peak_anchor_count: 2091
- history_days: 30
- min_history_days: 7
- enter_quantile: 0.95
- exit_quantile: 0.9
- merge_gap: 1
- min_duration: 3
- max_flank_hours: 168
- anchor_fallback_window_hours: 6
- warmup_hours(raw): -1
- warmup_hours_effective: 168

## Event Source Distribution
- gradient_width: 2091

## Monthly Event Distribution (top months)
- 2023-03: 48
- 2021-12: 47
- 2022-01: 47
- 2021-03: 46
- 2024-01: 46
- 2024-03: 46
- 2021-04: 45
- 2021-01: 44
- 2021-02: 44
- 2023-01: 44
- 2022-03: 43
- 2023-02: 43
