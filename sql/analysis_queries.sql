-- DemandPulse analytical queries (SQLite).  Tables are created by src/demandpulse/db.py
-- Every block starts with "-- name: <id>" and is executed by db.run_query(<id>).

-- name: yearly_national_traffic
-- Passengers per calendar year across ALL city pairs, with YoY growth and recovery vs 2019.
WITH y AS (
    SELECT CAST(strftime('%Y', date) AS INT) AS year,
           COUNT(DISTINCT date)               AS months_reported,
           SUM(pax)                           AS pax
    FROM pairs_monthly
    GROUP BY year
)
SELECT year, months_reported, pax,
       ROUND(100.0 * (pax - LAG(pax) OVER (ORDER BY year)) / LAG(pax) OVER (ORDER BY year), 1) AS yoy_pct,
       ROUND(100.0 * pax / (SELECT pax FROM y WHERE year = 2019), 1)                         AS pct_of_2019
FROM y
ORDER BY year;

-- name: top_markets_ttm
-- Ten largest city-pair markets over the last 12 months, with share of national traffic.
WITH last_month AS (SELECT MAX(date) AS d FROM pairs_monthly),
ttm AS (
    SELECT market, SUM(pax) AS pax
    FROM pairs_monthly, last_month
    WHERE date > DATE(last_month.d, '-12 months')
    GROUP BY market
)
SELECT market, pax,
       ROUND(100.0 * pax / SUM(pax) OVER (), 2) AS share_pct,
       RANK() OVER (ORDER BY pax DESC)          AS rnk
FROM ttm
ORDER BY pax DESC
LIMIT 10;

-- name: busiest_cities_ttm
-- Airport-level traffic (arrivals + departures) for the last 12 months.
WITH last_month AS (SELECT MAX(date) AS d FROM city_monthly)
SELECT city, SUM(pax) AS pax,
       ROUND(100.0 * SUM(pax) / SUM(SUM(pax)) OVER (), 1) AS share_pct
FROM city_monthly, last_month
WHERE date > DATE(last_month.d, '-12 months')
GROUP BY city
ORDER BY pax DESC
LIMIT 10;

-- name: seasonality_index
-- Average month-of-year demand relative to the year's monthly mean (2023-2025, post-COVID).
WITH m AS (
    SELECT CAST(strftime('%Y', date) AS INT) AS year,
           CAST(strftime('%m', date) AS INT) AS month, SUM(pax) AS pax
    FROM pairs_monthly
    WHERE date BETWEEN '2023-01-01' AND '2025-12-01'
    GROUP BY year, month
),
idx AS (SELECT year, month, pax / AVG(pax) OVER (PARTITION BY year) AS idx FROM m)
SELECT month, ROUND(AVG(idx), 3) AS seasonal_index
FROM idx GROUP BY month ORDER BY month;

-- name: fastest_growing_markets
-- Markets ranked by growth of the last 12 months versus the 12 months before (min. 500k pax).
-- Goa is excluded: DGCA's Dabolim/Mopa reporting change in Jan-2026 doubles its pairs (not real growth).
WITH last_month AS (SELECT MAX(date) AS d FROM pairs_monthly),
agg AS (
    SELECT market,
           SUM(CASE WHEN date >  DATE(l.d, '-12 months') THEN pax END) AS ttm,
           SUM(CASE WHEN date <= DATE(l.d, '-12 months')
                     AND date >  DATE(l.d, '-24 months') THEN pax END) AS prior
    FROM pairs_monthly, last_month l GROUP BY market
)
SELECT market, ttm, prior, ROUND(100.0 * (ttm - prior) / prior, 1) AS growth_pct,
       DENSE_RANK() OVER (ORDER BY 1.0 * (ttm - prior) / prior DESC) AS growth_rank
FROM agg WHERE prior > 500000 AND ttm > 500000 AND market NOT LIKE '%GOA%'
ORDER BY growth_pct DESC LIMIT 10;

-- name: airline_share_ttm
-- Scheduled domestic carriers: passengers and share over the last 12 months.
WITH last_month AS (SELECT MAX(date) AS d FROM carrier_monthly),
t AS (
    SELECT airline, SUM(pax) AS pax
    FROM carrier_monthly, last_month
    WHERE date > DATE(last_month.d, '-12 months') AND airline <> 'Total Domestic'
    GROUP BY airline
)
SELECT airline, pax, ROUND(100.0 * pax / SUM(pax) OVER (), 1) AS share_pct
FROM t WHERE pax > 0 ORDER BY pax DESC LIMIT 8;

-- name: directional_imbalance
-- Markets where one direction carries noticeably more passengers over the last 12 months.
WITH last_month AS (SELECT MAX(date) AS d FROM pairs_monthly)
SELECT market,
       SUM(pax_a_to_b) AS a_to_b, SUM(pax_b_to_a) AS b_to_a,
       ROUND(100.0 * (SUM(pax_a_to_b) - SUM(pax_b_to_a)) / SUM(pax), 2) AS imbalance_pct
FROM pairs_monthly, last_month
WHERE date > DATE(last_month.d, '-12 months')
GROUP BY market HAVING SUM(pax) > 500000
ORDER BY ABS(imbalance_pct) DESC LIMIT 10;

-- name: recovery_vs_2019_by_market
-- 2025 traffic as a % of 2019 for the biggest 2019 markets (COVID recovery scorecard).
WITH y AS (
    SELECT market, CAST(strftime('%Y', date) AS INT) AS year, SUM(pax) AS pax
    FROM pairs_monthly WHERE strftime('%Y', date) IN ('2019', '2025') GROUP BY market, year
)
SELECT a.market, a.pax AS pax_2019, b.pax AS pax_2025,
       ROUND(100.0 * b.pax / a.pax, 1) AS pct_of_2019
FROM y a JOIN y b ON a.market = b.market AND a.year = 2019 AND b.year = 2025
WHERE a.pax > 800000
ORDER BY a.pax DESC LIMIT 10;
