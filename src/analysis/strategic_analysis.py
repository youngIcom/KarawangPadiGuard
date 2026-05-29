"""
Generate reproducible strategic-analysis artifacts for KarawangPadiGuard.

Outputs:
- data/processed/top_district_value_at_risk.csv
- models/xgboost_risk_prediction_v1_feature_importance.csv
- reports/strategic_analysis_summary.md
"""

import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PRODUCTION_PATH = BASE_DIR / "data" / "processed" / "dataset_produksi_padi_karawang_cleaned.csv"
WEATHER_PATH = BASE_DIR / "data" / "processed" / "weather_data.csv"
FEATURES_PATH = BASE_DIR / "models" / "xgboost_risk_prediction_v1_features.json"
MODEL_PATH = BASE_DIR / "models" / "xgboost_risk_prediction_v1.pkl"
METRICS_PATH = BASE_DIR / "models" / "xgboost_risk_prediction_v1_metrics.json"
OUTPUT_VAR_PATH = BASE_DIR / "data" / "processed" / "top_district_value_at_risk.csv"
OUTPUT_IMPORTANCE_PATH = BASE_DIR / "models" / "xgboost_risk_prediction_v1_feature_importance.csv"
REPORT_PATH = BASE_DIR / "reports" / "strategic_analysis_summary.md"

RICE_PRICE_PER_TON_IDR = 6_000_000
LOSS_RATE_LOW = 0.20
LOSS_RATE_HIGH = 0.40
PREVENTABLE_RATE_LOW = 0.05
PREVENTABLE_RATE_HIGH = 0.10


def format_idr(value):
    if value >= 1_000_000_000_000:
        return f"Rp {value / 1_000_000_000_000:.2f} triliun"
    if value >= 1_000_000_000:
        return f"Rp {value / 1_000_000_000:.1f} miliar"
    if value >= 1_000_000:
        return f"Rp {value / 1_000_000:.1f} juta"
    return f"Rp {value:,.0f}"


def build_value_at_risk():
    production = pd.read_csv(PRODUCTION_PATH)
    production_2021 = production[production["tahun"] == 2021].copy()
    district = (
        production_2021.groupby("nama_kecamatan", as_index=False)["produksi_padi"]
        .sum()
        .sort_values("produksi_padi", ascending=False)
    )
    district["asset_value_idr"] = district["produksi_padi"] * RICE_PRICE_PER_TON_IDR
    district["loss_at_20pct_idr"] = district["asset_value_idr"] * LOSS_RATE_LOW
    district["loss_at_40pct_idr"] = district["asset_value_idr"] * LOSS_RATE_HIGH
    district["preventable_5pct_idr"] = district["asset_value_idr"] * PREVENTABLE_RATE_LOW
    district["preventable_10pct_idr"] = district["asset_value_idr"] * PREVENTABLE_RATE_HIGH
    district.to_csv(OUTPUT_VAR_PATH, index=False)
    return production_2021, district


def build_feature_importance():
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURES_PATH.read_text())
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_IMPORTANCE_PATH, index=False)
    return importance


def build_humidity_signal():
    weather = pd.read_csv(WEATHER_PATH)
    daily = (
        weather.groupby("date", as_index=False)
        .agg({
            "temperature": "mean",
            "humidity": "mean",
            "rainfall": "sum",
            "wind_speed": "mean",
            "cloud_cover": "mean",
        })
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["humidity_rolling_7"] = daily["humidity"].rolling(7).mean()
    daily["blast_favorable"] = (
        daily["temperature"].between(25, 28) & (daily["humidity"] >= 90)
    )
    daily["golden_window_signal"] = daily["humidity_rolling_7"] > 85
    signal_days = int(daily["golden_window_signal"].sum())
    blast_days = int(daily["blast_favorable"].sum())
    signal_and_blast_days = int((daily["golden_window_signal"] & daily["blast_favorable"]).sum())
    return daily, signal_days, blast_days, signal_and_blast_days


def write_report(production_2021, district, importance, signal_days, blast_days, signal_and_blast_days):
    metrics = json.loads(METRICS_PATH.read_text())
    accuracy = metrics["metrics"]["accuracy"] * 100
    f1_score = metrics["metrics"]["f1_score"] * 100
    total_production = production_2021["produksi_padi"].sum()
    total_asset = total_production * RICE_PRICE_PER_TON_IDR
    total_loss_low = total_asset * LOSS_RATE_LOW
    total_loss_high = total_asset * LOSS_RATE_HIGH
    preventable_low = total_asset * PREVENTABLE_RATE_LOW
    preventable_high = total_asset * PREVENTABLE_RATE_HIGH

    top_districts = district.head(5)
    top_district_lines = "\n".join(
        f"- {row.nama_kecamatan}: {row.produksi_padi:,.0f} ton, asset {format_idr(row.asset_value_idr)}"
        for row in top_districts.itertuples()
    )
    top_features = importance.head(10)
    top_feature_lines = "\n".join(
        f"- {row.feature}: {row.importance:.4f}"
        for row in top_features.itertuples()
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(f"""# Strategic Analysis Summary\n\nGenerated from local project artifacts.\n\n## Model Evidence\n\n- Risk model accuracy: {accuracy:.2f}%\n- Risk model F1-score: {f1_score:.2f}%\n- Feature set: {len(json.loads(FEATURES_PATH.read_text()))} weather, temporal, and vegetation-index features\n\n## Value at Risk\n\n- 2021 production rows: {len(production_2021)} villages\n- Total 2021 production: {total_production:,.0f} ton\n- Protected asset estimate: {format_idr(total_asset)}\n- Potential loss range at 20-40% yield loss: {format_idr(total_loss_low)} - {format_idr(total_loss_high)}\n- Preventable value range at 5-10% avoided loss: {format_idr(preventable_low)} - {format_idr(preventable_high)}\n\n## Highest Value Districts\n\n{top_district_lines}\n\n## Golden Window Signal\n\n- Days with 7-day rolling humidity above 85%: {signal_days}\n- Days directly favorable for Blast based on temperature and humidity: {blast_days}\n- Overlap days between both signals: {signal_and_blast_days}\n- Interpretation: rolling humidity is an early warning feature, while direct favorable days are a near-term escalation signal.\n\n## Top Model Features\n\n{top_feature_lines}\n\n## Generated Files\n\n- `{OUTPUT_VAR_PATH.relative_to(BASE_DIR)}`\n- `{OUTPUT_IMPORTANCE_PATH.relative_to(BASE_DIR)}`\n- `{REPORT_PATH.relative_to(BASE_DIR)}`\n""")


def main():
    OUTPUT_VAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    production_2021, district = build_value_at_risk()
    importance = build_feature_importance()
    _, signal_days, blast_days, signal_and_blast_days = build_humidity_signal()
    write_report(production_2021, district, importance, signal_days, blast_days, signal_and_blast_days)
    print(f"Wrote {OUTPUT_VAR_PATH.relative_to(BASE_DIR)}")
    print(f"Wrote {OUTPUT_IMPORTANCE_PATH.relative_to(BASE_DIR)}")
    print(f"Wrote {REPORT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
