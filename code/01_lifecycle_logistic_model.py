"""
Logistic lifecycle modeling for lithography patent evolution.

This script fits a three-parameter Logistic growth model to annual granted
patent counts, identifies lifecycle thresholds at 10%, 50%, and 90% of the
estimated saturation level, and saves fitted values, parameters, and a figure.

Example:
    python code/01_lifecycle_logistic_model.py \
        --input data/annual_patent_counts.csv \
        --output-dir outputs/lifecycle \
        --forecast-end-year 2028
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def logistic_model(t: np.ndarray, K: float, a: float, t0: float) -> np.ndarray:
    """Three-parameter Logistic growth model."""
    return K / (1.0 + np.exp(-a * (t - t0)))


def infer_columns(data: pd.DataFrame, year_column: str | None, count_column: str | None) -> Tuple[str, str]:
    """Infer year and annual count columns when they are not explicitly provided."""
    if year_column and count_column:
        return year_column, count_column

    normalized = {c.lower().strip().replace(" ", "_"): c for c in data.columns}

    year_candidates = ["year", "grant_year", "authorized_year", "publication_year"]
    count_candidates = [
        "patent_count",
        "granted_patents",
        "granted_patent_count",
        "count",
        "annual_count",
        "number_of_patents",
    ]

    inferred_year = year_column
    inferred_count = count_column

    if inferred_year is None:
        for name in year_candidates:
            if name in normalized:
                inferred_year = normalized[name]
                break

    if inferred_count is None:
        for name in count_candidates:
            if name in normalized:
                inferred_count = normalized[name]
                break

    if inferred_year is None or inferred_count is None:
        numeric_columns = list(data.select_dtypes(include=["number"]).columns)
        if inferred_year is None and len(numeric_columns) >= 1:
            inferred_year = numeric_columns[0]
        if inferred_count is None and len(numeric_columns) >= 2:
            inferred_count = numeric_columns[1]

    if inferred_year is None or inferred_count is None:
        raise ValueError(
            "Could not infer the year and patent-count columns. "
            "Please specify --year-column and --count-column."
        )

    return inferred_year, inferred_count


def load_annual_counts(input_path: Path, year_column: str | None, count_column: str | None) -> pd.DataFrame:
    """Load annual patent counts from a CSV or Excel file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(input_path)
    else:
        data = pd.read_csv(input_path)

    year_col, count_col = infer_columns(data, year_column, count_column)

    counts = data[[year_col, count_col]].copy()
    counts.columns = ["year", "annual_count"]
    counts["year"] = counts["year"].astype(int)
    counts["annual_count"] = pd.to_numeric(counts["annual_count"], errors="coerce").fillna(0).astype(float)
    counts = counts.groupby("year", as_index=False)["annual_count"].sum()
    counts = counts.sort_values("year")

    full_years = pd.DataFrame({"year": np.arange(counts["year"].min(), counts["year"].max() + 1)})
    counts = full_years.merge(counts, on="year", how="left").fillna({"annual_count": 0})
    return counts


def fit_logistic(counts: pd.DataFrame, k_lower: float | None, k_upper: float | None) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the Logistic model to cumulative annual counts."""
    start_year = int(counts["year"].min())
    t_actual = counts["year"].to_numpy() - start_year
    cumulative_actual = counts["annual_count"].cumsum().to_numpy()

    final_cumulative = float(cumulative_actual[-1])
    p0 = [final_cumulative * 1.1, 0.04, max(1, len(t_actual) * 0.6)]

    lower = [k_lower if k_lower is not None else final_cumulative * 0.90, 0.001, 0]
    upper = [k_upper if k_upper is not None else final_cumulative * 1.30, 1.000, len(t_actual) * 1.5]

    params, covariance = curve_fit(
        logistic_model,
        t_actual,
        cumulative_actual,
        p0=p0,
        bounds=(lower, upper),
        maxfev=30000,
    )
    return params, covariance


def lifecycle_threshold_years(start_year: int, K: float, a: float, t0: float) -> dict:
    """Calculate the years corresponding to 10%, 50%, and 90% of saturation."""
    t_10 = t0 - np.log(9) / a
    t_50 = t0
    t_90 = t0 + np.log(9) / a
    return {
        "year_10_percent": float(start_year + t_10),
        "year_50_percent": float(start_year + t_50),
        "year_90_percent": float(start_year + t_90),
        "t_10_percent": float(t_10),
        "t_50_percent": float(t_50),
        "t_90_percent": float(t_90),
    }


def save_outputs(
    counts: pd.DataFrame,
    params: np.ndarray,
    forecast_end_year: int,
    output_dir: Path,
    figure_name: str,
) -> None:
    """Save fitted values, model parameters, and lifecycle figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    K, a, t0 = [float(x) for x in params]
    start_year = int(counts["year"].min())
    last_actual_year = int(counts["year"].max())
    plot_years = np.arange(start_year, forecast_end_year + 1)
    t_plot = plot_years - start_year
    cumulative_fit = logistic_model(t_plot, K, a, t0)

    actual = counts.set_index("year")["annual_count"]
    fit_df = pd.DataFrame(
        {
            "year": plot_years,
            "annual_count": [float(actual.get(y, np.nan)) for y in plot_years],
            "cumulative_actual": [
                float(counts.loc[counts["year"] <= y, "annual_count"].sum()) if y <= last_actual_year else np.nan
                for y in plot_years
            ],
            "cumulative_fitted": cumulative_fit,
            "data_type": ["actual" if y <= last_actual_year else "forecast" for y in plot_years],
        }
    )
    fit_df.to_csv(output_dir / "lifecycle_fitted_values.csv", index=False)

    thresholds = lifecycle_threshold_years(start_year, K, a, t0)
    metadata = {
        "model": "N(t) = K / (1 + exp(-a * (t - t0)))",
        "start_year": start_year,
        "last_actual_year": last_actual_year,
        "forecast_end_year": int(forecast_end_year),
        "K": K,
        "a": a,
        "t0": t0,
        "actual_cumulative_patents": float(counts["annual_count"].sum()),
        **thresholds,
    }
    with open(output_dir / "lifecycle_logistic_parameters.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    create_lifecycle_figure(
        counts=counts,
        plot_years=plot_years,
        cumulative_fit=cumulative_fit,
        parameters=metadata,
        output_path=output_dir / figure_name,
    )

    print("Logistic lifecycle modeling completed.")
    print(f"K = {K:.2f}, a = {a:.4f}, t0 = {t0:.2f}")
    print(f"10% threshold year: {thresholds['year_10_percent']:.2f}")
    print(f"50% threshold year: {thresholds['year_50_percent']:.2f}")
    print(f"90% threshold year: {thresholds['year_90_percent']:.2f}")
    print(f"Outputs saved to: {output_dir}")


def create_lifecycle_figure(
    counts: pd.DataFrame,
    plot_years: np.ndarray,
    cumulative_fit: np.ndarray,
    parameters: dict,
    output_path: Path,
) -> None:
    """Create and save the Logistic lifecycle figure."""
    K = parameters["K"]
    actual_years = counts["year"].to_numpy()
    cumulative_actual = counts["annual_count"].cumsum().to_numpy()
    last_actual_year = parameters["last_actual_year"]

    predicted_mask = plot_years > last_actual_year
    actual_fit_mask = plot_years <= last_actual_year

    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(12, 7))

    plt.plot(
        plot_years[actual_fit_mask],
        cumulative_fit[actual_fit_mask],
        linewidth=2.5,
        label=f"Fitted ({actual_years.min()}-{last_actual_year})",
    )
    if predicted_mask.any():
        plt.plot(
            plot_years[predicted_mask],
            cumulative_fit[predicted_mask],
            linestyle="--",
            linewidth=2.5,
            label=f"Predicted ({plot_years[predicted_mask].min()}-{plot_years[predicted_mask].max()})",
        )

    plt.scatter(actual_years, cumulative_actual, alpha=0.6, s=20, label="Observed cumulative patents")
    plt.scatter(actual_years.min(), cumulative_actual[0], s=90, marker="*", zorder=10)

    year_10 = parameters["year_10_percent"]
    year_50 = parameters["year_50_percent"]

    plt.scatter(year_10, 0.1 * K, s=70, zorder=5, label="10% saturation")
    plt.scatter(year_50, 0.5 * K, s=70, zorder=5, label="50% saturation")
    plt.axvline(year_10, linestyle="--", linewidth=1.2)
    plt.axvline(year_50, linestyle="--", linewidth=1.2)

    plt.text((actual_years.min() + year_10) / 2, K * 0.82, "Embryonic", fontsize=12, ha="center", fontweight="bold")
    plt.text((year_10 + year_50) / 2, K * 0.82, "Growth", fontsize=12, ha="center", fontweight="bold")
    plt.text((year_50 + plot_years.max()) / 2, K * 0.82, "Mature", fontsize=12, ha="center", fontweight="bold")

    parameter_text = (
        f"Saturation (K): {K:.0f}\n"
        f"Actual cumulative ({actual_years.min()}-{last_actual_year}): {parameters['actual_cumulative_patents']:.0f}\n"
        f"Growth rate (a): {parameters['a']:.4f}"
    )
    plt.text(
        0.02,
        0.98,
        parameter_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.xticks(np.arange(actual_years.min(), plot_years.max() + 1, 5), rotation=45, fontsize=10)
    plt.xlabel("Year", fontsize=12, fontweight="bold")
    plt.ylabel("Cumulative number of granted patents", fontsize=12, fontweight="bold")
    plt.title("Lithography Technology Lifecycle Based on Logistic Modeling", fontsize=14, fontweight="bold", pad=18)
    plt.xlim(actual_years.min() - 1, plot_years.max() + 1)
    plt.ylim(0, K * 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a Logistic lifecycle model to annual granted patent counts.")
    parser.add_argument("--input", required=True, help="Path to a CSV or Excel file containing annual patent counts.")
    parser.add_argument("--output-dir", default="outputs/lifecycle", help="Directory for fitted values, parameters, and figure.")
    parser.add_argument("--year-column", default=None, help="Name of the year column. If omitted, the script attempts to infer it.")
    parser.add_argument("--count-column", default=None, help="Name of the annual patent-count column. If omitted, the script attempts to infer it.")
    parser.add_argument("--forecast-end-year", type=int, default=2028, help="Final year for fitted/predicted lifecycle values.")
    parser.add_argument("--k-lower", type=float, default=None, help="Lower bound for the saturation parameter K.")
    parser.add_argument("--k-upper", type=float, default=None, help="Upper bound for the saturation parameter K.")
    parser.add_argument("--figure-name", default="lithography_lifecycle_logistic.png", help="Output figure filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    counts = load_annual_counts(input_path, args.year_column, args.count_column)
    if args.forecast_end_year < int(counts["year"].max()):
        raise ValueError("--forecast-end-year must be greater than or equal to the last observed year.")

    params, _ = fit_logistic(counts, args.k_lower, args.k_upper)
    save_outputs(
        counts=counts,
        params=params,
        forecast_end_year=args.forecast_end_year,
        output_dir=output_dir,
        figure_name=args.figure_name,
    )


if __name__ == "__main__":
    main()
