# WLEL Data Quality Report (v1)

- Generated at: 2026-04-03T20:41:21
- Time range: 2021-01-01 00:00:00 -> 2025-11-27 16:00:00
- Total rows: 43001

## Data Cleaning
- Duplicate timestamps removed: 0
- Missing/inserted points filled: 10
- Irregular interval ratio (after unification): 0.000000

## Data Quality
- Missing rate (value): 0.000000
- Outlier rate (robust z): 0.004698
- Legacy peak density (`is_peak`): 0.070998
- Event window density (`is_peak_event`): 0.324318

## Event Summary
- Event count: 3053
- Event density per 1k points: 70.9983
- Duration mean/p50/p90/max: 4.568 / 5.000 / 5.000 / 5
- Apex mean/p90/max: 5881.716 / 7975.570 / 10541.681

## Threshold Config (locked for v1)
- event_mode: gradient_width
- boundary_semantics: threshold_crossing_hysteresis
- merge_policy: gap_based
- merge_gap_effective: 1
- peak_anchor_count: 3053
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
- gradient_width: 3053

## Monthly Event Distribution (top months)
- 2021-01: 62
- 2021-03: 62
- 2022-05: 62
- 2023-03: 62
- 2024-03: 62
- 2022-03: 61
- 2023-10: 61
- 2023-12: 61
- 2024-01: 61
- 2024-12: 61
- 2025-01: 61
- 2025-03: 61
