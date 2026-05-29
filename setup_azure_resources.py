"""Create or update the Azure ML workspace for KarawangPadiGuard.

Prerequisites:
1. Run `az login`.
2. Create the resource group first, for example:
   `az group create --name KarawangPadiGuard_RG --location southeastasia`
3. Install dependencies from `requirements_azure.txt`.
"""

import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential


def setup_workspace(subscription_id, resource_group, workspace_name, location):
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group)

    workspace = Workspace(
        name=workspace_name,
        location=location,
        display_name="KarawangPadiGuard ML Workspace",
        description=(
            "Workspace for KarawangPadiGuard experiment tracking, "
            "model artifacts, and future MLOps pipelines."
        ),
    )

    print(f"Creating/updating workspace: {workspace_name} in {location}...")
    ml_client.workspaces.begin_create(workspace).result()
    print(f"Workspace {workspace_name} is ready.")


if __name__ == "__main__":
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "KarawangPadiGuard_RG")
    workspace_name = os.environ.get("AZURE_ML_WORKSPACE", "karawangpadiguard-ml")
    location = os.environ.get("AZURE_LOCATION", "southeastasia")

    if not subscription_id:
        raise SystemExit("Set AZURE_SUBSCRIPTION_ID first.")

    setup_workspace(subscription_id, resource_group, workspace_name, location)
