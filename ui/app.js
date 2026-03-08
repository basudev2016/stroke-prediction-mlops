/**
 * app.js — Stroke Prediction MLOps Dashboard
 *
 * Tabs: Predict (active) | Pipeline | Observe | Monitor | Results
 * Future weeks will unlock disabled tabs.
 */

const API_BASE = window.location.origin;

// ──────────────────────────────────────────────
// Tab Navigation
// ──────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
        if (tab.classList.contains("disabled")) return;

        // Deactivate all
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

        // Activate clicked
        tab.classList.add("active");
        const target = document.getElementById(tab.dataset.tab);
        if (target) target.classList.add("active");
    });
});

// ──────────────────────────────────────────────
// Load model info on page load
// ──────────────────────────────────────────────
async function loadModelInfo() {
    const badge = document.getElementById("model-badge");
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();

        if (data.status === "healthy") {
            badge.textContent = `Model v${data.model_version} — ${data.model_name}`;
            badge.className = "badge healthy";
        } else {
            badge.textContent = "Model not loaded";
            badge.className = "badge error";
        }
    } catch (e) {
        badge.textContent = "API unreachable";
        badge.className = "badge error";
    }
}

// ──────────────────────────────────────────────
// Single Prediction
// ──────────────────────────────────────────────
async function predict() {
    const btn = document.getElementById("predict-btn");
    const card = document.getElementById("result-card");
    btn.disabled = true;
    btn.textContent = "Predicting...";

    const payload = {
        gender: document.getElementById("gender").value,
        age: parseFloat(document.getElementById("age").value),
        hypertension: parseInt(document.getElementById("hypertension").value),
        heart_disease: parseInt(document.getElementById("heart_disease").value),
        ever_married: document.getElementById("ever_married").value,
        work_type: document.getElementById("work_type").value,
        Residence_type: document.getElementById("Residence_type").value,
        avg_glucose_level: parseFloat(document.getElementById("avg_glucose_level").value),
        bmi: parseFloat(document.getElementById("bmi").value),
        smoking_status: document.getElementById("smoking_status").value
    };

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Prediction failed");
        }

        const data = await res.json();

        // Update result card
        const isStroke = data.prediction === 1;
        card.className = `result-card ${isStroke ? "stroke" : "no-stroke"}`;

        document.getElementById("result-icon").textContent = isStroke ? "⚠️" : "✅";
        document.getElementById("result-label").textContent = data.prediction_label;
        document.getElementById("result-confidence").textContent =
            `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById("result-model").textContent =
            `Model v${data.model_version}`;

        card.classList.remove("hidden");

    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = "🔍 Predict Stroke Risk";
    }
}

// ──────────────────────────────────────────────
// Batch Prediction
// ──────────────────────────────────────────────
async function predictBatch() {
    const fileInput = document.getElementById("batch-file");
    const resultDiv = document.getElementById("batch-result");

    if (!fileInput.files.length) {
        alert("Please select a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.classList.add("hidden");
    resultDiv.innerHTML = "Scoring...";

    try {
        const res = await fetch(`${API_BASE}/predict/batch`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Batch prediction failed");
        }

        const data = await res.json();

        resultDiv.innerHTML = `
            <div class="stat">Total Records: <span class="stat-value">${data.total_records}</span></div>
            <div class="stat">Stroke Predicted: <span class="stat-value" style="color:#f87171">${data.stroke_count}</span></div>
            <div class="stat">No Stroke: <span class="stat-value" style="color:#4ade80">${data.no_stroke_count}</span></div>
            <div class="stat">Model: <span class="stat-value">v${data.model_version}</span></div>
        `;
        resultDiv.classList.remove("hidden");

    } catch (e) {
        resultDiv.innerHTML = `<span style="color:#f87171">Error: ${e.message}</span>`;
        resultDiv.classList.remove("hidden");
    }
}

// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
loadModelInfo();