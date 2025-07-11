document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const uploadArea = document.getElementById("uploadArea");
    const fileInfo = document.getElementById("fileInfo");
    const fileNameDisplay = document.getElementById("fileName");
    const removeFile = document.getElementById("removeFile");
    const predictBtn = document.getElementById("predictBtn");
    const progressBar = document.getElementById("progressBar");
    const progressFill = progressBar.querySelector(".progress-fill");
    const result = document.getElementById("result");
    const emotionName = document.getElementById("emotionName");
    const confidenceText = document.getElementById("confidenceText");
    const emotionIcon = document.getElementById("emotionIcon");
    const confidenceFill = document.getElementById("confidenceFill");
    const modelInfo = document.getElementById("modelInfo");
    const recentList = document.getElementById("recentList");
    const loadLogsBtn = document.getElementById("loadLogsBtn");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const API_URL = "http://127.0.0.1:8000";

    // Handle file selection
    uploadArea.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            uploadArea.style.display = "none";
            fileInfo.style.display = "flex";
            fileNameDisplay.textContent = file.name;
            predictBtn.disabled = false;
        }
    });

    removeFile.addEventListener("click", () => {
        fileInput.value = "";
        fileInfo.style.display = "none";
        uploadArea.style.display = "block";
        predictBtn.disabled = true;
    });

    // Predict emotion using the API
    predictBtn.addEventListener("click", async () => {
        if (!fileInput.files[0]) return;

        // Show progress
        progressBar.style.display = "block";
        progressFill.style.width = "0%";

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                showResult(data);
            } else {
                alert(`Prediction failed: ${response.status}`);
            }
        } catch (error) {
            console.error("Prediction error:", error);
            alert("Failed to predict emotion.");
        } finally {
            progressBar.style.display = "none";
        }
    });

    function showResult(data) {
        const { emotion, confidence, model } = data;

        // Update result display
        result.style.display = "block";
        emotionName.textContent = emotion;
        confidenceText.textContent = `${(confidence * 100).toFixed(2)}%`;
        modelInfo.textContent = `Model: ${model}`;

        // Update confidence
        confidenceFill.style.width = `${(confidence * 100).toFixed(2)}%`;
    }

    // Load recent logs
    loadLogsBtn.addEventListener("click", async () => {
        try {
            const response = await fetch(`${API_URL}/logs?limit=5`);
            if (response.ok) {
                const data = await response.json();
                updateRecentList(data.logs);
            } else {
                alert(`Failed to load logs: ${response.status}`);
            }
        } catch (error) {
            console.error("Load logs error:", error);
            alert("Failed to load recent logs.");
        }
    });

    function updateRecentList(logs) {
        recentList.innerHTML = "";
        if (logs.length === 0) {
            recentList.innerHTML = "<p class='no-data'>No recent analysis</p>";
            return;
        }

        logs.forEach((log) => {
            const listItem = document.createElement("div");
            listItem.classList.add("recent-item");
            listItem.innerHTML = `
                <p>${log.timestamp} - ${log.emotion} (${(log.confidence * 100).toFixed(2)}%)</p>
            `;
            recentList.appendChild(listItem);
        });
    }

    // Check API status
    async function checkAPIStatus() {
        try {
            const response = await fetch(`${API_URL}/health`);
            if (response.ok) {
                statusDot.style.backgroundColor = "#4caf50";
                statusText.textContent = "Connected";
            } else {
                statusDot.style.backgroundColor = "#f44336";
                statusText.textContent = "Error";
            }
        } catch (error) {
            statusDot.style.backgroundColor = "#f44336";
            statusText.textContent = "Disconnected";
        }
    }

    checkAPIStatus();
});

