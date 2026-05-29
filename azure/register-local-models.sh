#!/usr/bin/env bash
set -euo pipefail

: "${AZURE_RESOURCE_GROUP:?Set AZURE_RESOURCE_GROUP}"
: "${AZURE_ML_WORKSPACE:?Set AZURE_ML_WORKSPACE}"

RISK_MODEL_PATH="models/xgboost_risk_prediction_v1.pkl"
CV_MODEL_PATH="models/mobilenetv3_rice_disease_v1_best.keras"

if [[ ! -f "$RISK_MODEL_PATH" ]]; then
  echo "Missing risk model: $RISK_MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$CV_MODEL_PATH" ]]; then
  echo "Missing CV model: $CV_MODEL_PATH" >&2
  exit 1
fi

echo "Registering XGBoost risk model..."
az ml model create   --resource-group "$AZURE_RESOURCE_GROUP"   --workspace-name "$AZURE_ML_WORKSPACE"   --name xgboost-risk-prediction   --version 1   --type custom_model   --path "$RISK_MODEL_PATH"

echo "Registering MobileNetV3 rice disease model..."
az ml model create   --resource-group "$AZURE_RESOURCE_GROUP"   --workspace-name "$AZURE_ML_WORKSPACE"   --name mobilenetv3-rice-disease   --version 1   --type custom_model   --path "$CV_MODEL_PATH"

echo "Registered models:"
az ml model list   --resource-group "$AZURE_RESOURCE_GROUP"   --workspace-name "$AZURE_ML_WORKSPACE"   --query "[].{name:name, version:version, type:type}"   --output table
