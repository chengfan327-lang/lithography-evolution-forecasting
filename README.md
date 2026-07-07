# Data and Code for Lithography Technology Evolution and Forecasting

This repository provides the data, code, and figures supporting the manuscript:

**Dormancy and Technological Leap in Lithography: A Three-Dimensional Indicator Framework for Evolutionary Analysis and Forecasting**

submitted to *Technological Forecasting and Social Change*.

## Overview

This study investigates the long-term technological evolution and future trajectories of lithography technologies based on global granted lithography patents from 1918 to 2023.

The manuscript develops an integrated analytical framework combining:

- Logistic technology lifecycle identification;
- Sentence-BERT-KMeans technological theme identification;
- S–G–M technological theme classification;
- Time-series forecasting and robustness validation.

This repository provides reproducible code, aggregated datasets, derived indicators, IPC co-occurrence network data, and figures used to support the main empirical results reported in the manuscript.

## Repository structure

```text
lithography-evolution-forecasting/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── code/
│   ├── 01_lifecycle_logistic_model.py
│   ├── 02_topic_model.py
│   └── 03_forecasting_results.py
│
├── data/
│   ├── Granted patents in lithography.csv
│   ├── ipc_statistics.csv
│   ├── topic_keywords.csv
│   └── ipc_network/
│       ├── Embryonic (1918-1980)_core_edges.csv
│       ├── Embryonic (1918-1980)_core_nodes.csv
│       ├── Growth (1981-2008)_core_edges.csv
│       ├── Growth (1981-2008)_core_nodes.csv
│       ├── Maturity (2009-2023)_core_edges.csv
│       └── Maturity (2009-2023)_core_nodes.csv
│
└── figures/
    ├── Theme Coherence Curve.png
    ├── Two-Dimensional Clustering Visualization.png
    ├── ipc_statistics-Embryonic (1918-1980).png
    ├── ipc_statistics-Growth (1981-2008).png
    ├── ipc_statistics-Maturity (2009-2023).png
    └── Evolution trend results/
```

## Data description

The `data/` folder contains aggregated and derived datasets used in the empirical analysis.

- `Granted patents in lithography.csv`: Annual counts of granted lithography patents used for Logistic lifecycle modeling.
- `ipc_statistics.csv`: IPC-based statistical results used to describe the technological distribution of lithography patents.
- `topic_keywords.csv`: Identified technological themes and representative keywords generated from the topic modeling process.
- `ipc_network/`: Stage-specific node and edge files used for IPC co-occurrence network analysis across lifecycle stages.

The `ipc_network/` folder includes the following files:

- `Embryonic (1918-1980)_core_nodes.csv`
- `Embryonic (1918-1980)_core_edges.csv`
- `Growth (1981-2008)_core_nodes.csv`
- `Growth (1981-2008)_core_edges.csv`
- `Maturity (2009-2023)_core_nodes.csv`
- `Maturity (2009-2023)_core_edges.csv`

These files support the IPC co-occurrence network analysis reported in the manuscript.

## Code description

The `code/` folder contains three main scripts corresponding to the empirical workflow of the manuscript.

### 1. Lifecycle identification

```text
01_lifecycle_logistic_model.py
```

This script fits a three-parameter Logistic model to annual granted patent counts and identifies lifecycle thresholds based on cumulative technological growth.

Main outputs include:

- estimated Logistic parameters;
- fitted lifecycle curve;
- lifecycle threshold years;
- lifecycle visualization.

Example:

```bash
python code/01_lifecycle_logistic_model.py \
  --input "data/Granted patents in lithography.csv" \
  --output-dir figures/lifecycle \
  --forecast-end-year 2028
```

If the column names in the input file are not `year` and `patent_count`, please specify the corresponding column names:

```bash
python code/01_lifecycle_logistic_model.py \
  --input "data/Granted patents in lithography.csv" \
  --year-column Year \
  --count-column Granted_patents \
  --output-dir figures/lifecycle \
  --forecast-end-year 2028
```

### 2. Technological theme identification

```text
02_topic_model.py
```

This script identifies technological themes from patent titles and abstracts using Sentence-BERT embeddings, UMAP dimensionality reduction, K-Means clustering, and BERTopic-based topic representation.

The script is designed for reproducible research while avoiding the redistribution of licensed raw patent records.

Example:

```bash
python code/02_topic_model.py \
  --input data/lithography_cleaned.xlsx \
  --model-path /path/to/PatentSBERTa \
  --output-dir figures/topic_model \
  --n-topics 17
```

Note: The raw patent title and abstract file is not publicly redistributed due to database licensing restrictions. Users with legitimate access to licensed patent data may run this script using their own patent text data.

### 3. Forecasting and robustness validation

```text
03_forecasting_results.py
```

This script supports the construction of topic-level time-series data and forecasting robustness validation.

Example for building topic-level time series:

```bash
python code/03_forecasting_results.py build-timeseries \
  --documents data/final_documents_with_topics.xlsx \
  --topics data/topic_info.csv \
  --output data/topic_year_timeseries.xlsx
```

Example for validating the time-series data:

```bash
python code/03_forecasting_results.py validate \
  --input data/topic_year_timeseries.xlsx \
  --output-dir figures/forecasting_validation
```

Example for UCM-based forecasting validation:

```bash
python code/03_forecasting_results.py forecast-ucm \
  --input data/topic_year_timeseries.xlsx \
  --output data/ucm_forecast_results.xlsx \
  --metrics Patent_Count Total_Citation \
  --forecast-until-year 2030 \
  --drop-last-n-years 2
```

Note: Some intermediate files required for forecasting, such as document-level topic assignment files, are not publicly redistributed if they contain licensed patent text fields. Users with legitimate access to the original patent data may reproduce these files using the topic modeling script.

## Figure description

The `figures/` folder contains the main figures and visual outputs used in the manuscript, including:

- topic coherence curve;
- two-dimensional clustering visualization;
- IPC co-occurrence network figures across lifecycle stages;
- evolutionary trend results.

The folder `figures/Evolution trend results/` contains additional visual outputs related to technological theme evolution and forecasting.

## Software requirements

The analysis was conducted using Python. Required packages are listed in `requirements.txt`.

To install the required packages, run:

```bash
pip install -r requirements.txt
```

The required packages include:

```text
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
scipy>=1.9.0
torch>=2.0.0
bertopic>=0.16.0
sentence-transformers>=2.2.0
umap-learn>=0.5.3
scikit-learn>=1.2.0
safetensors>=0.3.0
statsmodels>=0.14.0
openpyxl>=3.1.0
```

## Data availability and restrictions

The raw patent records were retrieved from the Incopat global patent database. Due to database licensing restrictions, the complete raw patent records, including original patent titles, abstracts, claims, and DWPI fields, cannot be publicly redistributed.

To support reproducibility, this repository provides aggregated and derived datasets, IPC network data, figures, and analysis code used in the manuscript.

Users with legitimate access to the original patent database may reproduce the text-mining procedure using their own licensed patent records and the scripts provided in this repository.

## Reproducibility notes

The main empirical workflow is organized as follows:

1. Use `01_lifecycle_logistic_model.py` to identify lifecycle stages based on annual granted patent counts.
2. Use `02_topic_model.py` to identify technological themes from patent text data.
3. Use derived theme-level data to calculate technological share (S), growth intensity (G), and technological impact (M).
4. Use `03_forecasting_results.py` to construct topic-level time-series data and conduct forecasting robustness validation.
5. Use the data and figures in this repository to support the lifecycle-based theme classification and future trend analysis reported in the manuscript.

Some file paths and column names may need to be adjusted according to the local data format.

## Citation

If you use this repository, please cite the corresponding article after publication.

## License

The code in this repository is released for academic and reproducible research purposes. The redistributed datasets are aggregated or derived data and do not include licensed raw patent records.
