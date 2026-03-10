# ──────────────────────────────────────────────
# main.tf — Provider configuration and API enablement
# ──────────────────────────────────────────────
# This file tells Terraform:
#   - We're using Google Cloud
#   - Which project and region to target
#   - Which APIs to enable

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required GCP APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",              # Cloud Run
    "artifactregistry.googleapis.com", # Docker image storage
    "cloudbuild.googleapis.com",       # Building containers
    "monitoring.googleapis.com",       # Metrics and dashboards
    "logging.googleapis.com",          # Structured logs
  ])

  project = var.project_id
  service = each.value

  disable_on_destroy = false
}