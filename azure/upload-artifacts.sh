#!/usr/bin/env bash
set -euo pipefail

: "${AZURE_STORAGE_ACCOUNT:?Set AZURE_STORAGE_ACCOUNT}"
: "${AZURE_STORAGE_CONTAINER:=karawangpadiguard}"

az storage container create \
  --account-name "$AZURE_STORAGE_ACCOUNT" \
  --name "$AZURE_STORAGE_CONTAINER" \
  --auth-mode login

az storage blob upload-batch \
  --account-name "$AZURE_STORAGE_ACCOUNT" \
  --destination "$AZURE_STORAGE_CONTAINER/data/processed" \
  --source data/processed \
  --auth-mode login \
  --overwrite

az storage blob upload-batch \
  --account-name "$AZURE_STORAGE_ACCOUNT" \
  --destination "$AZURE_STORAGE_CONTAINER/models" \
  --source models \
  --auth-mode login \
  --overwrite
