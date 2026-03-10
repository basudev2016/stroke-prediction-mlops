# ──────────────────────────────────────────────
# outputs.tf — What Terraform reports after apply
# ──────────────────────────────────────────────

output "api_url" {
  description = "Cloud Run API URL"
  value       = google_cloud_run_v2_service.api.uri
}

output "incoming_bucket" {
  description = "GCS bucket for incoming patient data"
  value       = google_storage_bucket.incoming.name
}

output "scored_bucket" {
  description = "GCS bucket for scored Excel output"
  value       = google_storage_bucket.scored.name
}

output "reports_bucket" {
  description = "GCS bucket for Evidently drift reports"
  value       = google_storage_bucket.reports.name
}

output "docker_repo" {
  description = "Artifact Registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}