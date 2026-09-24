# DemandPulse – key findings (auto-generated from the real DGCA data)

**Data:** 134 months (2015-04 → 2026-07), 155 cities, 1498 city pairs. City-pair totals reconcile with DGCA's own carrier totals (median gap 0.000%, max 0.63%).

## Demand

- Passengers per calendar year (millions): 2016: 99.5, 2017: 116.8, 2018: 138.7, 2019: 143.7, 2020: 62.9, 2021: 85.0, 2022: 123.2, 2023: 152.0, 2024: 161.0, 2025: 166.9 (months DGCA never published are estimated: 2021-05, 2024-05).
- 2025 traffic = 116.1% of 2019.
- Last 12 months: 167.5 M passengers (+0.4% vs the 12 months before).
- Top 10 city pairs carry 20.3% of traffic; 48 pairs make up half of it.
- Five busiest airports (DELHI, MUMBAI, BENGALURU, HYDERABAD, KOLKATA) handle 52.4% of airport traffic.
- Seasonality: peak Dec (index 1.08), trough Jul (index 0.94).

## Airlines

- Share of domestic passengers (TTM): IndiGo 64.2%, Air India 14.8%, Air India Express 11.5%, Akasa Air 5.3%.
- Average scheduled-domestic passenger load factor (TTM): 85.1%.

## Forecast accuracy (rolling-origin backtest, WAPE %)

Selected model: **xgb**.

| Model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| ensemble | 6.6 | 9.0 | 9.9 | 12.1 |
| ets | 7.2 | 9.8 | 11.6 | 14.5 |
| snaive | 10.9 | 10.9 | 11.0 | 10.8 |
| snaive_drift | 11.0 | 12.7 | 14.7 | 16.3 |
| xgb | 6.7 | 9.3 | 9.8 | 10.8 |

Skill vs seasonal naive (%): h=1: +38.7, h=3: +15.1, h=6: +10.7, h=12: +0.4

80% interval coverage (leave-one-origin-out): h=1: 77.4%, h=3: 78.5%, h=6: 78.8%, h=12: 79.2%

## Opportunity ranking

1. BAGDOGRA – DELHI – score 76.6 (Emerging star)
2. BENGALURU – DELHI – score 70.7 (Growth engine)
3. BENGALURU – PUNE – score 70.2 (Core trunk)
4. CHENNAI – KOLKATA – score 70.2 (Emerging star)
5. DELHI – KOLKATA – score 69.2 (Core trunk)
