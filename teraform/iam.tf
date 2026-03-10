# ──────────────────────────────────────────────
# iam.tf — Service accounts and permissions
# ──────────────────────────────────────────────

# Service account for GitHub Actions deployment
resource "google_service_account" "github_deploy" {
  account_id   = "github-deploy"
  display_name = "GitHub Deploy"
  description  = "Used by GitHub Actions to deploy to Cloud Run"
}

# Permission: Deploy to Cloud Run
resource "google_project_iam_member" "github_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.github_deploy.email}"
}

# Permission: Push Docker images
resource "google_project_iam_member" "github_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.github_deploy.email}"
}

# Permission: Act as service account
resource "google_project_iam_member" "github_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.github_deploy.email}"
}