# Data sources

| File | What it is | Origin |
|---|---|---|
| `raw/dgca_domestic_city_pairs.csv` | Monthly domestic passengers per city pair, both directions (Apr-2015 → Jul-2026) | DGCA (India) *Monthly Statistics – Domestic Air Transport*, consolidated by [Vonter/india-aviation-traffic](https://github.com/Vonter/india-aviation-traffic) → `aggregated/domestic/city.csv` |
| `raw/dgca_domestic_carriers.csv` | Monthly carrier statistics: passengers, seat-km, passenger-km, load factor, fleet | same repository → `aggregated/domestic/carrier.csv` |
| `raw/LICENSE_ODbL.txt` | Open Database Licence that covers the consolidated data | same repository |

Refresh with `python run_pipeline.py --refresh-data` (downloads the latest CSVs from GitHub, then rebuilds everything).

Derived files (`processed/`, `demandpulse.db`, `../outputs/`) are produced by the pipeline. Because they are derived
from ODbL data, share them under the same licence and credit DGCA and the repository above.
