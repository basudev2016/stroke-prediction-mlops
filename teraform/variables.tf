# ──────────────────────────────────────────────
# variables.tf — Input variables
# ──────────────────────────────────────────────

variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "stroke-mlops"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "asia-south1"
}

variable "alert_email" {
  description = "Email for monitoring alerts"
  type        = string
  default     = "basudevpanda@gmail.com"
}