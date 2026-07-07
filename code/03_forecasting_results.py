"""
UCM-based forecasting and robustness validation for lithography technology themes.

This script supports the forecasting analysis in the lithography technology evolution
study. It can (1) construct annual topic-level time series from document-topic results,
(2) run pre-modeling validation checks, and (3) generate Unobserved Components Model
(UCM) forecasts for selected indicators such as patent counts and citation counts.

The script is designed for reproducible research using aggregated or derived datasets.
Raw patent records from licensed databases should not be redistributed through a public
repository.

Example:
    python code/03_forecasting_results.py \
        --input data/topic_year_timeseries.xlsx \
        --output-dir outputs/forecasting \
        --metrics Patent_Count Total_Citation \
        --forecast-until-year 2030 \
        --drop-last-n-years 2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


REQUIRED_TIME_SERIES_COLUMNS = [
    "Topic_ID",
    "Topic_Name",
    "Year",
    "Patent_Count",
    "Avg_Confidence",
    "Total_Citation",
    "Avg_Citation",
    "Share_S",
    "Growth_G",
    "Impact_M",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file based on its file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}. Use CSV or Excel files.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a dataframe to CSV or Excel based on its file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}. Use CSV or Excel files.")


def extract_year(series: pd.Series) -> pd.Series:
    """Extract four-digit years from a text or numeric year column."""
    years = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(years, errors="coerce")


def build_topic_year_timeseries(
    documents_path: str | Path,
    topics_path: str | Path,
    output_path: str | Path,
    topic_col: str = "Topic_ID",
    time_col: str = "Time",
    confidence_col: str = "Confidence",
    citation_col: str = "Citation",
    topic_info_topic_col: str = "Topic",
    topic_info_name_col: str = "Name",
) -> pd.DataFrame:
    """Construct annual topic-level indicators from document-topic assignments.

    Expected document-level fields are topic ID, time/year, confidence, and citation
    count. The output contains annual patent count, mean confidence, citation statistics,
    technological share, growth intensity, and relative citation impact.
    """
    docs = read_table(documents_path)
    topics = read_table(topics_path)

    required_doc_cols = [topic_col, time_col, confidence_col, citation_col]
    missing_doc_cols = [col for col in required_doc_cols if col not in docs.columns]
    if missing_doc_cols:
        raise ValueError(f"Missing required document columns: {missing_doc_cols}")

    required_topic_cols = [topic_info_topic_col, topic_info_name_col]
    missing_topic_cols = [col for col in required_topic_cols if col not in topics.columns]
    if missing_topic_cols:
        raise ValueError(f"Missing required topic-info columns: {missing_topic_cols}")

    docs = docs.copy()
    docs["Year"] = extract_year(docs[time_col])
    docs = docs.dropna(subset=["Year"]).copy()
    docs["Year"] = docs["Year"].astype(int)
    docs[citation_col] = pd.to_numeric(docs[citation_col], errors="coerce").fillna(0)
    docs[confidence_col] = pd.to_numeric(docs[confidence_col], errors="coerce").fillna(0)

    grouped = (
        docs.groupby([topic_col, "Year"])
        .agg(
            Patent_Count=(topic_col, "count"),
            Avg_Confidence=(confidence_col, "mean"),
            Total_Citation=(citation_col, "sum"),
            Avg_Citation=(citation_col, "mean"),
        )
        .reset_index()
        .rename(columns={topic_col: "Topic_ID"})
    )

    min_year = int(docs["Year"].min())
    max_year = int(docs["Year"].max())
    all_topics = sorted(docs[topic_col].dropna().unique())
    full_index = pd.MultiIndex.from_product(
        [all_topics, range(min_year, max_year + 1)], names=["Topic_ID", "Year"]
    )
    grouped = grouped.set_index(["Topic_ID", "Year"]).reindex(full_index).reset_index()

    fill_zero_cols = ["Patent_Count", "Total_Citation", "Avg_Confidence", "Avg_Citation"]
    grouped[fill_zero_cols] = grouped[fill_zero_cols].fillna(0)

    topic_info = topics[[topic_info_topic_col, topic_info_name_col]].rename(
        columns={topic_info_topic_col: "Topic_ID", topic_info_name_col: "Topic_Name"}
    )
    merged = grouped.merge(topic_info, on="Topic_ID", how="left")

    yearly_totals = merged.groupby("Year")["Patent_Count"].sum().rename("Yearly_Total")
    merged = merged.merge(yearly_totals.reset_index(), on="Year", how="left")
    merged["Share_S"] = np.where(
        merged["Yearly_Total"] > 0,
        merged["Patent_Count"] / merged["Yearly_Total"],
        0,
    )

    merged = merged.sort_values(["Topic_ID", "Year"])
    merged["Prev_Count"] = merged.groupby("Topic_ID")["Patent_Count"].shift(1)
    denominator = merged["Prev_Count"].where(merged["Prev_Count"] != 0, 1)
    merged["Growth_G"] = np.where(
        merged["Prev_Count"].isna(),
        0,
        (merged["Patent_Count"] - merged["Prev_Count"]) / denominator,
    )

    yearly_avg_citation = docs.groupby("Year")[citation_col].mean().rename("Yearly_Global_Avg_Cite")
    merged = merged.merge(yearly_avg_citation.reset_index(), on="Year", how="left")
    merged["Impact_M"] = np.where(
        merged["Yearly_Global_Avg_Cite"] > 0,
        merged["Avg_Citation"] / merged["Yearly_Global_Avg_Cite"],
        0,
    )

    final_df = merged[REQUIRED_TIME_SERIES_COLUMNS].copy()
    final_df = final_df.sort_values(["Topic_ID", "Year"])

    float_cols = ["Avg_Confidence", "Avg_Citation", "Share_S", "Growth_G", "Impact_M"]
    final_df[float_cols] = final_df[float_cols].round(4)

    write_table(final_df, output_path)
    LOGGER.info("Topic-year time series exported to %s", output_path)
    return final_df


def validate_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run basic validation checks for topic-year time series data."""
    missing_columns = [col for col in REQUIRED_TIME_SERIES_COLUMNS if col not in df.columns]
    issues: dict[str, object] = {"missing_columns": missing_columns}

    if missing_columns:
        return pd.DataFrame(), issues

    working = df.copy()
    working["Year"] = pd.to_numeric(working["Year"], errors="coerce")
    duplicate_rows = working[working.duplicated(subset=["Topic_ID", "Year"], keep=False)]
    issues["duplicate_topic_year_rows"] = int(len(duplicate_rows))

    anomaly_report = {}
    for col in ["Patent_Count", "Share_S", "Impact_M"]:
        values = pd.to_numeric(working[col], errors="coerce")
        anomaly_report[col] = {
            "missing": int(values.isna().sum()),
            "infinite": int(np.isinf(values.dropna()).sum()),
            "negative": int((values.dropna() < 0).sum()),
        }
    issues["indicator_anomalies"] = anomaly_report

    summary_rows = []
    for topic_id, group in working.groupby("Topic_ID"):
        years = sorted(group["Year"].dropna().astype(int).tolist())
        if not years:
            continue
        expected_years = list(range(min(years), max(years) + 1))
        missing_years = sorted(set(expected_years) - set(years))
        is_sorted = years == group["Year"].dropna().astype(int).tolist()
        summary_rows.append(
            {
                "Topic_ID": topic_id,
                "Year_Range": f"{min(years)}-{max(years)}",
                "Record_Count": len(years),
                "Is_Sorted": bool(is_sorted),
                "Is_Continuous": len(missing_years) == 0,
                "Missing_Years": ";".join(map(str, missing_years)) if missing_years else "",
            }
        )

    summary = pd.DataFrame(summary_rows)
    issues["non_continuous_topics"] = summary.loc[
        summary.get("Is_Continuous", pd.Series(dtype=bool)) == False, "Topic_ID"
    ].tolist() if not summary.empty else []

    avg_citation_missing = int(working["Avg_Citation"].isna().sum())
    issues["avg_citation_missing"] = avg_citation_missing
    issues["avg_citation_handling_note"] = (
        "If Avg_Citation is missing because no patents exist for a topic-year, set it to 0. "
        "If citation data are unavailable despite existing patents, inspect the raw citation field."
    )

    return summary, issues


def safe_filename(text: object) -> str:
    """Create a filesystem-safe filename component."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:120] or "untitled"


def choose_tick_step(year_count: int) -> int:
    """Choose a readable major tick interval for annual plots."""
    if year_count <= 15:
        return 1
    if year_count <= 30:
        return 2
    if year_count <= 60:
        return 5
    if year_count <= 120:
        return 10
    return 20


def forecast_with_ucm(
    input_path: str | Path,
    output_path: str | Path,
    metrics: Sequence[str],
    forecast_until_year: int = 2030,
    drop_last_n_years: int = 2,
    confidence_level: float = 0.80,
    plots_dir: str | Path | None = None,
    save_plots: bool = True,
    show_plots: bool = False,
    max_history_year: int | None = None,
) -> pd.DataFrame:
    """Forecast selected topic-level indicators using an Unobserved Components Model.

    The model uses a local linear trend specification and produces point forecasts and
    confidence intervals. It is mainly intended as a robustness-validation model for
    technology trend forecasting.
    """
    df = read_table(input_path)
    missing_metrics = [metric for metric in metrics if metric not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found in input data: {missing_metrics}")

    required_cols = ["Topic_ID", "Topic_Name", "Year"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    if "Data_Type" in df.columns:
        df = df[df["Data_Type"].astype(str).str.lower() == "historical"].copy()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    if max_history_year is not None:
        df = df[df["Year"] <= max_history_year].copy()

    df = df.sort_values(["Topic_ID", "Year"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if plots_dir is None:
        plots_dir = output_path.parent / "forecast_plots"
    plots_dir = Path(plots_dir)
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    alpha = 1.0 - confidence_level
    results: list[dict[str, object]] = []

    for topic_id, topic_data in df.groupby("Topic_ID"):
        topic_data = topic_data.sort_values("Year").copy()
        topic_name = topic_data["Topic_Name"].iloc[0]

        if drop_last_n_years > 0 and len(topic_data) > drop_last_n_years:
            topic_data = topic_data.iloc[:-drop_last_n_years].copy()

        years_hist = topic_data["Year"].to_numpy(dtype=int)
        if len(years_hist) < 4:
            LOGGER.warning("Skipping Topic %s because fewer than four observations are available.", topic_id)
            continue

        last_hist_year = int(years_hist[-1])
        periods_to_predict = int(forecast_until_year - last_hist_year)
        if periods_to_predict <= 0:
            LOGGER.warning("Skipping Topic %s because forecast horizon is not beyond history.", topic_id)
            continue

        future_years = np.arange(last_hist_year + 1, forecast_until_year + 1)
        LOGGER.info("Forecasting Topic %s (%s)", topic_id, topic_name)

        for metric in metrics:
            y_hist = pd.to_numeric(topic_data[metric], errors="coerce").fillna(0).to_numpy(dtype=float)
            nobs = len(y_hist)

            try:
                model = sm.tsa.UnobservedComponents(y_hist, level="local linear trend")
                fitted_model = model.fit(disp=False)

                hist_prediction = fitted_model.get_prediction(start=0, end=nobs - 1)
                hist_fit = np.maximum(0, np.asarray(hist_prediction.predicted_mean, dtype=float))

                future_prediction = fitted_model.get_prediction(start=nobs, end=nobs + periods_to_predict - 1)
                future_mean = np.maximum(0, np.asarray(future_prediction.predicted_mean, dtype=float))
                confidence_interval = np.asarray(future_prediction.conf_int(alpha=alpha), dtype=float)
                future_lower = np.maximum(0, confidence_interval[:, 0])
                future_upper = np.maximum(0, confidence_interval[:, 1])
            except Exception as exc:
                LOGGER.warning("UCM failed for Topic %s, metric %s: %s", topic_id, metric, exc)
                continue

            integer_like = metric in {"Patent_Count", "Total_Citation"}
            if integer_like:
                hist_fit_out = np.rint(hist_fit).astype(int)
                future_mean_out = np.rint(future_mean).astype(int)
                future_lower_out = np.rint(future_lower).astype(int)
                future_upper_out = np.rint(future_upper).astype(int)
            else:
                hist_fit_out = np.round(hist_fit, 4)
                future_mean_out = np.round(future_mean, 4)
                future_lower_out = np.round(future_lower, 4)
                future_upper_out = np.round(future_upper, 4)

            for i, year in enumerate(years_hist):
                results.append(
                    {
                        "Topic_ID": topic_id,
                        "Topic_Name": topic_name,
                        "Metric": metric,
                        "Year": int(year),
                        "Actual_Value": y_hist[i],
                        "Predicted_Value": hist_fit_out[i],
                        "Conf_Lower": hist_fit_out[i],
                        "Conf_Upper": hist_fit_out[i],
                        "Confidence_Level": confidence_level,
                        "Data_Type": "Historical",
                    }
                )

            for i, year in enumerate(future_years):
                results.append(
                    {
                        "Topic_ID": topic_id,
                        "Topic_Name": topic_name,
                        "Metric": metric,
                        "Year": int(year),
                        "Actual_Value": np.nan,
                        "Predicted_Value": future_mean_out[i],
                        "Conf_Lower": future_lower_out[i],
                        "Conf_Upper": future_upper_out[i],
                        "Confidence_Level": confidence_level,
                        "Data_Type": "Forecast",
                    }
                )

            if save_plots or show_plots:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(years_hist, y_hist, label="Actual value", zorder=5)
                ax.plot(years_hist, hist_fit, linestyle="-", label="Historical fitted trend")

                plot_years = np.insert(future_years, 0, years_hist[-1])
                plot_mean = np.insert(future_mean, 0, hist_fit[-1])
                plot_lower = np.insert(future_lower, 0, hist_fit[-1])
                plot_upper = np.insert(future_upper, 0, hist_fit[-1])

                ax.plot(plot_years, plot_mean, linestyle="--", linewidth=2, label="Forecast trend")
                ax.fill_between(plot_years, plot_lower, plot_upper, alpha=0.2, label=f"{int(confidence_level * 100)}% confidence interval")

                clean_metric = metric.replace("_", " ")
                clean_topic_name = str(topic_name).replace("_", " ")
                ax.set_title(f"Forecast trend: {clean_topic_name}", fontsize=13)
                ax.set_xlabel("Year")
                ax.set_ylabel(clean_metric)
                ax.set_xlim(years_hist[0] - 1, forecast_until_year + 1)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(choose_tick_step(len(years_hist) + len(future_years))))
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.legend(loc="upper left")
                fig.tight_layout()

                if save_plots:
                    filename = f"Topic_{safe_filename(topic_id)}_{safe_filename(metric)}.png"
                    fig.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
                if show_plots:
                    plt.show()
                plt.close(fig)

    if not results:
        raise RuntimeError("No forecasting results were generated. Check input data and parameters.")

    final_df = pd.DataFrame(results)
    write_table(final_df, output_path)
    LOGGER.info("Forecasting results exported to %s", output_path)
    if save_plots:
        LOGGER.info("Forecast plots exported to %s", plots_dir)
    return final_df


def run_validation(input_path: str | Path, output_dir: str | Path) -> None:
    """Validate a topic-year time series file and export validation reports."""
    df = read_table(input_path)
    summary, issues = validate_time_series(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary.empty:
        summary.to_csv(output_dir / "timeseries_validation_summary.csv", index=False)
    with open(output_dir / "timeseries_validation_issues.json", "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    LOGGER.info("Validation reports exported to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, validate, and forecast topic-year time series for lithography technology themes."
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build-timeseries", help="Build topic-year time series from document-topic results.")
    build_parser.add_argument("--documents", required=True, help="Document-topic assignment file, CSV or Excel.")
    build_parser.add_argument("--topics", required=True, help="Topic information file, CSV or Excel.")
    build_parser.add_argument("--output", default="data/topic_year_timeseries.xlsx", help="Output topic-year time series file.")
    build_parser.add_argument("--topic-col", default="Topic_ID")
    build_parser.add_argument("--time-col", default="Time")
    build_parser.add_argument("--confidence-col", default="Confidence")
    build_parser.add_argument("--citation-col", default="Citation")
    build_parser.add_argument("--topic-info-topic-col", default="Topic")
    build_parser.add_argument("--topic-info-name-col", default="Name")

    validate_parser = subparsers.add_parser("validate", help="Validate a topic-year time series file.")
    validate_parser.add_argument("--input", required=True, help="Topic-year time series file, CSV or Excel.")
    validate_parser.add_argument("--output-dir", default="outputs/forecasting/validation")

    forecast_parser = subparsers.add_parser("forecast-ucm", help="Run UCM forecasts for selected metrics.")
    forecast_parser.add_argument("--input", required=True, help="Topic-year time series file, CSV or Excel.")
    forecast_parser.add_argument("--output", default="outputs/forecasting/ucm_forecast_results.xlsx")
    forecast_parser.add_argument("--metrics", nargs="+", default=["Patent_Count", "Total_Citation"])
    forecast_parser.add_argument("--forecast-until-year", type=int, default=2030)
    forecast_parser.add_argument("--drop-last-n-years", type=int, default=2)
    forecast_parser.add_argument("--confidence-level", type=float, default=0.80)
    forecast_parser.add_argument("--plots-dir", default=None)
    forecast_parser.add_argument("--max-history-year", type=int, default=None)
    forecast_parser.add_argument("--no-plots", action="store_true", help="Disable figure export.")
    forecast_parser.add_argument("--show-plots", action="store_true", help="Display plots interactively.")

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.command == "build-timeseries":
        build_topic_year_timeseries(
            documents_path=args.documents,
            topics_path=args.topics,
            output_path=args.output,
            topic_col=args.topic_col,
            time_col=args.time_col,
            confidence_col=args.confidence_col,
            citation_col=args.citation_col,
            topic_info_topic_col=args.topic_info_topic_col,
            topic_info_name_col=args.topic_info_name_col,
        )
    elif args.command == "validate":
        run_validation(input_path=args.input, output_dir=args.output_dir)
    elif args.command == "forecast-ucm":
        forecast_with_ucm(
            input_path=args.input,
            output_path=args.output,
            metrics=args.metrics,
            forecast_until_year=args.forecast_until_year,
            drop_last_n_years=args.drop_last_n_years,
            confidence_level=args.confidence_level,
            plots_dir=args.plots_dir,
            save_plots=not args.no_plots,
            show_plots=args.show_plots,
            max_history_year=args.max_history_year,
        )
    else:
        raise SystemExit("Please specify a command: build-timeseries, validate, or forecast-ucm.")


if __name__ == "__main__":
    main()
