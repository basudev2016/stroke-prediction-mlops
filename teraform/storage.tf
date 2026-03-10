# ──────────────────────────────────────────────
# storage.tf — GCS buckets and Artifact Registry
# ──────────────────────────────────────────────

# Docker image repository
resource "google_artifact_registry_repository" "docker" {
  repository_id = "stroke-mlops-repo"
  location      = var.region
  format        = "DOCKER"
  description   = "Stroke MLOps Docker images"

  depends_on = [google_project_service.apis]
}

# Bucket for incoming patient data (triggers inference)
resource "google_storage_bucket" "incoming" {
  name     = "${var.project_id}-incoming"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = true
}

# Bucket for scored Excel output
resource "google_storage_bucket" "scored" {
  name     = "${var.project_id}-scored"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = true
}

# Bucket for Evidently drift reports
resource "google_storage_bucket" "reports" {
  name     = "${var.project_id}-reports"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = true
}