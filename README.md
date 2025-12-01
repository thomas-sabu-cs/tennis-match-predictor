### Tennis Match Prediction Model

#### Overview

This project builds a machine learning model to predict ATP tennis match outcomes using historical match data from 2020–2024.

The core dataset comes from Jeff Sackmann’s open tennis data repositories on GitHub, which are widely used in the tennis analytics community.

#### Data Sources

- **ATP Match Results (2020–2024)**  
  - Source: Jeff Sackmann’s `tennis_atp` repository  
  - Files used:
    - `atp_matches_2020.csv`
    - `atp_matches_2021.csv`
    - `atp_matches_2022.csv`
    - `atp_matches_2023.csv`
    - `atp_matches_2024.csv`
  - Combined locally into:
    - `atp_matches_2020_2024.csv`

Each row represents a single completed ATP singles match.

#### Dataset Summary (2020–2024)

- **Total matches:** 13,174  
- **Years covered:** 2020–2024  
- **Surfaces:** Hard, Clay, Grass  
- **Tournament levels:** Grand Slam (G), Masters (M), ATP (A), Olympics (O), Davis Cup (D), Futures/others (F)  
- **Match statistics coverage:** ~94–98% of matches per year have full serve/return stats.

#### Important Columns

**Match context**
- `tourney_id` – tournament identifier  
- `tourney_name` – tournament name  
- `tourney_date` – start date (YYYYMMDD as integer)  
- `surface` – court surface (Hard/Clay/Grass)  
- `tourney_level` – tournament category (G, M, A, etc.)  
- `round` – match round (R32, QF, SF, F, etc.)  
- `best_of` – number of sets (3 or 5)

**Player information**
- `winner_name`, `loser_name` – player names  
- `winner_id`, `loser_id` – numeric player IDs  
- `winner_rank`, `loser_rank` – ATP ranking at match time  
- `winner_rank_points`, `loser_rank_points` – ATP ranking points  
- `winner_age`, `loser_age` – age in years  
- `winner_ht`, `loser_ht` – height in cm (may have missing values)  
- `winner_hand`, `loser_hand` – playing hand (‘R’, ‘L’, sometimes ‘U’ for unknown)  
- `winner_ioc`, `loser_ioc` – country code

**Score and duration**
- `score` – full match score as string (e.g. `6-2 7-6(4)`)  
- `minutes` – match duration in minutes (some missing)

**Serve/return statistics (winner side, prefix `w_`)**
- `w_ace` – aces  
- `w_df` – double faults  
- `w_svpt` – total service points  
- `w_1stIn` – first serves in  
- `w_1stWon` – points won on first serve  
- `w_2ndWon` – points won on second serve  
- `w_SvGms` – service games played  
- `w_bpSaved` – break points saved  
- `w_bpFaced` – break points faced  

**Serve/return statistics (loser side, prefix `l_`)**
- `l_ace`, `l_df`, `l_svpt`, `l_1stIn`, `l_1stWon`, `l_2ndWon`, `l_SvGms`, `l_bpSaved`, `l_bpFaced` – same definitions as above for the loser.

#### Target Definition

The primary prediction target is **match winner**.

To model this, we typically transform each row into a **player-pair representation**, e.g.:

- `player_A` vs `player_B` with:
  - Features as differences or ratios (rank difference, age difference, etc.)
  - Target: `1` if `player_A` is the winner, `0` otherwise

Or we can simply:
- Use the row as-is and treat `winner_name` as the positive class when constructing pairwise features.

#### Planned Features

Example engineered features (per match):

- **Ranking-based:**
  - `rank_diff = loser_rank - winner_rank`
  - `rank_points_diff = winner_rank_points - loser_rank_points`

- **Demographics:**
  - `age_diff = winner_age - loser_age`
  - `height_diff = winner_ht - loser_ht`

- **Surface & context:**
  - One-hot encoded `surface`
  - One-hot encoded `tourney_level`
  - One-hot encoded `round`
  - `is_best_of_5` from `best_of`

- **Serve/return strength (per match, can later aggregate per player):**
  - Winner 1st serve percentage: `w_1stIn / w_svpt`
  - Winner 1st serve points won %: `w_1stWon / w_1stIn`
  - Winner 2nd serve points won %: `w_2ndWon / (w_svpt - w_1stIn)`
  - Break point save rate: `w_bpSaved / w_bpFaced` (when `w_bpFaced > 0`)
  - Similar metrics for loser.

Later, we may compute **rolling / historical** stats per player (e.g., last 10 matches on clay).

#### Modeling Approach

1. **Data cleaning**
   - Handle missing heights, ages, and match stats.
   - Filter to matches with sufficient feature coverage.
   - Convert `tourney_date` into a proper date and use it for time-based splits.

2. **Train/validation split**
   - Time-based: e.g., train on 2020–2023, validate/test on 2024 to mimic real prediction.

3. **Models to try**
   - Logistic Regression (baseline)
   - Random Forest / Gradient Boosted Trees (e.g., XGBoost, LightGBM)
   - Simple neural network (optional, after baselines)

4. **Evaluation metrics**
   - Accuracy
   - Log loss
   - Brier score
   - Calibration (how well predicted probabilities match actual outcomes)

#### Files in This Project

- `atp_matches_2020_2024.csv` – combined raw ATP matches (2020–2024)  
- `atp_matches_2024.csv` – standalone 2024 ATP matches  
- `README.md` – project description (this file)  
- (Planned)
  - `notebooks/01_exploration.ipynb` – exploratory data analysis  
  - `notebooks/02_feature_engineering.ipynb` – feature creation  
  - `notebooks/03_modeling.ipynb` – training and evaluation  
  - `src/` – reusable preprocessing and modeling scripts

#### How to Recreate the Dataset

1. Install Python and dependencies (example):

```bash
pip install pandas requests
```

2. Download and combine yearly files:

```python
import pandas as pd
import requests
from io import StringIO

years = [2020, 2021, 2022, 2023, 2024]
dfs = []

for year in years:
    url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
    df = pd.read_csv(url)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("atp_matches_2020_2024.csv", index=False)
```

#### Next Steps

- Implement a preprocessing script to:
  - Build player-pair features
  - Split data into train/validation/test by date
- Train baseline models and log performance.
- Iterate on more advanced features (player form, surface-specific stats, head-to-head, etc.).
