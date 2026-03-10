# ──────────────────────────────────────────────
# cloud_run.tf — API serving on Cloud Run
# ──────────────────────────────────────────────
# Deploys the FastAPI + UI as a serverless container
# Scales to zero when idle — no cost when not in use

resource "google_cloud_run_v2_service" "api" {
  name     = "stroke-api"
  location = var.region

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/stroke-mlops-repo/stroke-api:latest"

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          memory = "1Gi"
          cpu    = "1"
        }
      }
    }

    scaling {
      min_instance_count = 0  # Scale to zero — no cost when idle
      max_instance_count = 1  # Limit for cost control
    }
  }

  depends_on = [google_project_service.apis]
}

# Allow public access (unauthenticated)
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}