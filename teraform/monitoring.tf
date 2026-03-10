# ──────────────────────────────────────────────
# monitoring.tf — Dashboards, alerts, notifications
# ──────────────────────────────────────────────

# Email notification channel for alerts
resource "google_monitoring_notification_channel" "email" {
  display_name = "Workshop Alert Email"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

# Alert: High error rate on Cloud Run API
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "API Error Rate > 5%"
  combiner     = "OR"

  conditions {
    display_name = "Error rate exceeds 5%"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class!=\"2xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s"
  }
}

# Alert: High latency on Cloud Run API
resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "API p95 Latency > 2s"
  combiner     = "OR"

  conditions {
    display_name = "Latency exceeds 2 seconds"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 2000

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s"
  }
}