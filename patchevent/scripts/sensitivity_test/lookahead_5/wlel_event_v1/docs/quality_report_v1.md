# WLEL Data Quality Report (v1)

- Generated at: 2026-04-03T20:41:25
- Time range: 2021-01-01 00:00:00 -> 2025-11-27 16:00:00
- Total rows: 43001

## Data Cleaning
- Duplicate timestamps removed: 0
- Missing/inserted points filled: 10
- Irregular interval ratio (after unification): 0.000000

## Data Quality
- Missing rate (value): 0.000000
- Outlier rate (robust z): 0.004698
- Legacy peak density (`is_peak`): 0.059533
- Event window density (`is_peak_event`): 0.280807

## Event Summary
- Event count: 2560
- Event density per 1k points: 59.5335
- Duration mean/p50/p90/max: 4.717 / 5.000 / 5.000 / 5
- Apex mean/p90/max: 5896.915 / 7921.482 / 10541.681

## Threshold Config (locked for v1)
- event_mode: gradient_width
- boundary_semantics: threshold_crossing_hysteresis
- merge_policy: gap_based
- merge_gap_effective: 1
- peak_anchor_count: 2560
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
- gradient_width: 2560

## Monthly Event Distribution (top months)
- 2023-03: 62
- 2021-03: 61
- 2021-04: 60
- 2021-11: 58
- 2021-12: 58
- 2024-03: 58
- 2022-10: 57
- 2022-01: 56
- 2022-03: 56
- 2022-11: 56
- 2024-01: 56
- 2021-01: 55
