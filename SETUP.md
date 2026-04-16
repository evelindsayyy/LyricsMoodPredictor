# Setup

Quick notes for running this from a fresh clone.

## 1. Python environment

Python 3.10+ is what I've been using. A venv keeps things tidy:

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the dataset

The raw data isn't in the repo — the main file is ~256MB which is over GitHub's limit. Grab SpotGenTrack from one of:

- Kaggle mirror (easier): https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset
- Mendeley original: https://data.mendeley.com/datasets/4m2x4zngny/1

Unzip it so that the file lives at:

```
SpotGenTrack/Data Sources/spotify_tracks.csv
```

(at the project root). That's the only file the notebooks use — the other CSVs in the bundle aren't needed.

## 3. Regenerate the processed data

The processed dataset `data/processed/songs_labeled.csv` is also not committed (it's derived from the raw file and is also too big for GitHub). To regenerate it, run the first notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

and run all cells. Or from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

Later notebooks expect `data/processed/songs_labeled.csv` to exist.
