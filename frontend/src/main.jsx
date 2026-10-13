import { StrictMode, useEffect, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const rawApiUrl = import.meta.env.VITE_API_URL;
const API_URL = rawApiUrl
  ? rawApiUrl.replace(/\/$/, "")
  : (import.meta.env.DEV ? "" : "http://127.0.0.1:8000");

const VEHICLE_CLASSES = [
  "Motorized 2-wheeler", "Ambassador taxi", "Autorickshaw", "Bicycle",
  "Bus", "Car", "Minitruck", "Motorvan", "Rickshaw", "Toto (E-rickshaw)", "Truck", "Van",
];

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function App() {
  const [backendStatus, setBackendStatus] = useState("checking"); // 'checking' | 'ready' | 'error'
  const [mode, setMode] = useState("image");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const [retryTrigger, setRetryTrigger] = useState(0);

  const ANALYSIS_STAGES = [
    { label: "Uploading media", icon: "📤", duration: 3000 },
    { label: "Running YOLOv8 detection", icon: "🔍", duration: 12000 },
    { label: "Classifying vehicle types", icon: "🧠", duration: 10000 },
    { label: "Rendering annotated output", icon: "🎨", duration: 5000 },
  ];

  // Cycle through analysis stages while analyzing
  useEffect(() => {
    if (!isAnalyzing) {
      setAnalysisStage(0);
      return;
    }
    if (analysisStage >= ANALYSIS_STAGES.length - 1) return;

    const timer = setTimeout(() => {
      setAnalysisStage((s) => Math.min(s + 1, ANALYSIS_STAGES.length - 1));
    }, ANALYSIS_STAGES[analysisStage].duration);

    return () => clearTimeout(timer);
  }, [isAnalyzing, analysisStage]);

  const fileInputRef = useRef(null);
  const acceptTypes = mode === "image" ? "image/png,image/jpeg,image/webp" : "video/mp4,video/quicktime,video/x-msvideo";

  // Health check on mount and retry
  useEffect(() => {
    let isCancelled = false;
    let pollTimeout;

    async function checkHealth() {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 6000);
        let res;
        const endpoint = API_URL ? `${API_URL}/health` : "/health";
        try {
          res = await fetch(endpoint, { signal: controller.signal, cache: "no-store" });
        } catch (initialErr) {
          if (!API_URL || API_URL.includes("localhost")) {
            res = await fetch("http://127.0.0.1:8000/health", { signal: controller.signal, cache: "no-store" });
          } else {
            throw initialErr;
          }
        }
        clearTimeout(timer);

        if (res && res.ok) {
          const data = await res.json();
          if (data.status === "ok") {
            if (!isCancelled) setBackendStatus("ready");
            return;
          } else if (data.status === "error") {
            if (!isCancelled) {
              setBackendStatus("error");
              setErrorMessage(data.detail || "Model initialization failed on backend.");
            }
            return;
          }
        }
      } catch {
        // Backend might still be spinning up
      }

      if (!isCancelled) {
        pollTimeout = setTimeout(checkHealth, 3500);
      }
    }

    setBackendStatus("checking");
    checkHealth();

    return () => {
      isCancelled = true;
      clearTimeout(pollTimeout);
    };
  }, [retryTrigger]);

  // Handle file preview cleanup
  useEffect(() => {
    if (!file || mode !== "image") {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file, mode]);

  function handleModeChange(nextMode) {
    if (nextMode === mode) return;
    setMode(nextMode);
    setFile(null);
    setResult(null);
    setErrorMessage("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleFileSelected(selectedFile) {
    if (!selectedFile) return;
    setFile(selectedFile);
    setResult(null);
    setErrorMessage("");
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  }

  function clearFile() {
    setFile(null);
    setResult(null);
    setErrorMessage("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) {
      setErrorMessage("Please select or drop a media file to inspect.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const endpoint = API_URL ? `${API_URL}/api/analyze/${mode}` : `/api/analyze/${mode}`;
      let response;
      try {
        response = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });
      } catch (networkErr) {
        if (!API_URL || API_URL.includes("localhost")) {
          response = await fetch(`http://127.0.0.1:8000/api/analyze/${mode}`, {
            method: "POST",
            body: formData,
          });
        } else {
          throw networkErr;
        }
      }

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const errorDetail = data?.detail || `Server responded with status ${response.status}`;
        throw new Error(errorDetail);
      }

      setResult(data);
    } catch (err) {
      const isConnectionError = err.name === "TypeError" || err.message.includes("Failed to fetch");
      if (isConnectionError) {
        const target = API_URL || "http://127.0.0.1:8000";
        const isRender = target.includes("onrender.com");
        setErrorMessage(
          isRender
            ? `Unable to reach Render backend at ${target}. Free-tier instances take 45-60s to wake up.`
            : `Unable to connect to local backend at ${target}. Please verify uvicorn is running.`
        );
        setBackendStatus("checking");
      } else {
        setErrorMessage(err.message || "An error occurred during analysis.");
      }
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="shell">
      {/* Blurred Backdrop Loading Overlay while Backend Wakes Up on Render */}
      {backendStatus !== "ready" && (
        <div className="loading-backdrop">
          <div className="loading-dialog">
            <div className="loading-spinner-ring" />
            <h2 className="loading-title">Connecting to Backend</h2>
            <p className="loading-description">
              The backend is deployed on Render. Free-tier services take around 45–60 seconds to wake up and load the ML models after being idle.
            </p>
            <div className="loading-status-badge">
              <span className="loading-dot-pulse" />
              <span>Waking instance on Render...</span>
            </div>
            {backendStatus === "error" && (
              <button
                type="button"
                className="retry-connect-btn"
                onClick={() => setRetryTrigger((prev) => prev + 1)}
              >
                Retry Connection
              </button>
            )}
          </div>
        </div>
      )}

      {/* Navigation */}
      <header className="navbar">
        <div className="brand-group">
          <span className="brand-mark" />
          <span className="brand-title">Vehix</span>
          <span className="brand-tag">v1.0</span>
        </div>
      </header>

      {/* Hero & Upload */}
      <main className="hero-layout">
        <div className="hero-info">
          <div className="eyebrow">
            <span className="eyebrow-accent" />
            Computer Vision Pipeline
          </div>
          <h1 className="hero-heading">
            Indian road vehicle detection &amp; classification
          </h1>
          <p className="hero-description">
            A dual-stage vision architecture combining YOLOv8 for vehicle spatial localization with a dedicated classifier trained across 12 Indian vehicular categories.
          </p>

          <div className="taxonomy-preview">
            {VEHICLE_CLASSES.map((cls) => (
              <span key={cls} className="taxonomy-tag">
                {cls}
              </span>
            ))}
          </div>
        </div>

        {/* Upload Panel */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Input Media</span>
            <div className="segmented-control">
              <button
                type="button"
                className={`segmented-btn ${mode === "image" ? "active" : ""}`}
                onClick={() => handleModeChange("image")}
              >
                Image
              </button>
              <button
                type="button"
                className={`segmented-btn ${mode === "video" ? "active" : ""}`}
                onClick={() => handleModeChange("video")}
              >
                Video
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <div
              className={`dropzone ${isDragOver ? "drag-active" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
            >
              <svg className="dropzone-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <span className="dropzone-label">
                {isDragOver ? "Drop file to upload" : "Select or drag file here"}
              </span>
              <span className="dropzone-hint">
                {mode === "image" ? "JPEG, PNG or WEBP" : "MP4, MOV or AVI"}
              </span>
              <input
                ref={fileInputRef}
                type="file"
                accept={acceptTypes}
                onChange={(e) => handleFileSelected(e.target.files?.[0])}
              />
            </div>

            {file && (
              <div className="selected-file-card">
                <div className="file-info">
                  {previewUrl && (
                    <img src={previewUrl} alt="Thumbnail preview" className="file-preview-thumb" />
                  )}
                  <div className="file-meta-col">
                    <span className="file-name" title={file.name}>{file.name}</span>
                    <span className="file-size">{formatBytes(file.size)}</span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={clearFile}
                  className="remove-file-btn"
                  title="Remove file"
                  aria-label="Remove file"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
            )}

            {errorMessage && (
              <div className="alert-error">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <span>{errorMessage}</span>
              </div>
            )}

            {isAnalyzing ? (
              <div className="analysis-tracker">
                <div className="tracker-header">
                  <span className="tracker-spinner" />
                  <span className="tracker-title">Analyzing {mode}...</span>
                </div>
                <div className="tracker-stages">
                  {ANALYSIS_STAGES.map((stage, i) => (
                    <div
                      key={stage.label}
                      className={`tracker-stage ${
                        i < analysisStage ? "done" : i === analysisStage ? "active" : "pending"
                      }`}
                    >
                      <div className="stage-indicator">
                        {i < analysisStage ? (
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        ) : i === analysisStage ? (
                          <span className="stage-pulse" />
                        ) : (
                          <span className="stage-dot" />
                        )}
                      </div>
                      <span className="stage-icon">{stage.icon}</span>
                      <span className="stage-label">{stage.label}</span>
                    </div>
                  ))}
                </div>
                <div className="tracker-progress-bar">
                  <div
                    className="tracker-progress-fill"
                    style={{ width: `${((analysisStage + 1) / ANALYSIS_STAGES.length) * 100}%` }}
                  />
                </div>
              </div>
            ) : (
              <button
                type="submit"
                className="btn-submit"
                disabled={!file}
              >
                <span>Run Analysis</span>
              </button>
            )}
          </form>
        </div>
      </main>

      {/* Results View */}
      {result && <AnalysisReport result={result} />}

      {/* Methodology Section */}
      <section className="pipeline-section">
        <div className="pipeline-header">
          <h2 className="pipeline-title">Inference Architecture</h2>
          <p className="pipeline-subtitle">Pure PyTorch pipeline — TensorFlow-free, TorchScript-optimized, JIT-warmed at startup.</p>
        </div>

        <div className="pipeline-steps">
          <div className="pipeline-step">
            <div className="step-num">01 / INGESTION</div>
            <div className="step-title">Frame Acquisition</div>
            <p className="step-desc">Input is decoded via OpenCV and downscaled to max 480px for efficient processing.</p>
          </div>
          <div className="pipeline-step">
            <div className="step-num">02 / LOCALIZATION</div>
            <div className="step-title">YOLOv8n Detection</div>
            <p className="step-desc">Runs at 320px resolution under torch.inference_mode — detects vehicles with COCO class filtering.</p>
          </div>
          <div className="pipeline-step">
            <div className="step-num">03 / TAXONOMY</div>
            <div className="step-title">TorchScript Classifier</div>
            <p className="step-desc">Batched crops pass through a TorchScript-compiled MobileNetV2 + Multi-Head Attention model.</p>
          </div>
          <div className="pipeline-step">
            <div className="step-num">04 / MEMORY</div>
            <div className="step-title">Optimized Cleanup</div>
            <p className="step-desc">gc.collect + malloc_trim reclaims memory after every request to stay under 512 MB RAM.</p>
          </div>
          <div className="pipeline-step">
            <div className="step-num">05 / OUTPUT</div>
            <div className="step-title">Annotated Report</div>
            <p className="step-desc">Bounding boxes, class labels, and dual confidence scores are rendered and served.</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <span>Vehix — Indian Road Vehicle Detection &amp; Classification</span>
        <span>Built with FastAPI, PyTorch &amp; YOLOv8</span>
      </footer>
    </div>
  );
}

function AnalysisReport({ result }) {
  if (result.type === "video") {
    return (
      <section className="results-section">
        <div className="results-header">
          <h2 className="results-title">Processed Video Output</h2>
          <span className="results-meta">Sequence Analysis</span>
        </div>

        <div className="metrics-row">
          <div className="metric-card">
            <div className="metric-label">Frames Processed</div>
            <div className="metric-value">{result.processed_frames}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Total Frames</div>
            <div className="metric-value">{result.total_frames}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Effective FPS</div>
            <div className="metric-value">{result.fps}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Duration</div>
            <div className="metric-value">{result.elapsed_time}s</div>
          </div>
        </div>

        <div className="result-media-card">
          <div className="media-toolbar">
            <span>Annotated Video Stream</span>
            <a href={result.result_url} target="_blank" rel="noreferrer" className="media-open-link">
              Open Raw Video ↗
            </a>
          </div>
          <video controls src={result.result_url} preload="metadata" />
        </div>
      </section>
    );
  }

  const vehicles = result.vehicles || [];
  const topVehicle = vehicles[0]?.label || "None";
  const avgDetectionConf = vehicles.length
    ? Math.round(vehicles.reduce((acc, v) => acc + v.detection_confidence, 0) / vehicles.length)
    : 0;

  return (
    <section className="results-section">
      <div className="results-header">
        <h2 className="results-title">Analysis Output</h2>
        <span className="results-meta">{vehicles.length} Vehicle{vehicles.length === 1 ? "" : "s"} Identified</span>
      </div>

      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-label">Detected Count</div>
          <div className="metric-value">{vehicles.length}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Primary Class</div>
          <div className="metric-value" style={{ fontSize: "18px" }}>{topVehicle}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Avg. Localization</div>
          <div className="metric-value">{avgDetectionConf}%</div>
        </div>
      </div>

      <div className="results-grid">
        {/* Annotated Image */}
        <div className="result-media-card">
          <div className="media-toolbar">
            <span>Annotated Canvas</span>
            <a href={result.result_url} target="_blank" rel="noreferrer" className="media-open-link">
              Open Raw Image ↗
            </a>
          </div>
          <img src={result.result_url} alt="Annotated detection scene" />
        </div>

        {/* Vehicles Breakdown */}
        <div className="vehicles-panel">
          <div className="panel-subheading">Identified Crops ({vehicles.length})</div>
          {vehicles.length === 0 ? (
            <div className="vehicle-card">
              <span className="vehicle-name" style={{ color: "var(--text-muted)" }}>
                No vehicles detected in this image.
              </span>
            </div>
          ) : (
            vehicles.map((v) => (
              <div className="vehicle-card" key={`${v.number}-${v.crop_url}`}>
                <img src={v.crop_url} alt={v.label} className="vehicle-crop" />
                <div className="vehicle-details">
                  <div className="vehicle-title-row">
                    <span className="vehicle-name">{v.label}</span>
                    <span className="vehicle-index">#{v.number}</span>
                  </div>

                  <div className="confidence-group">
                    <div className="confidence-row">
                      <span className="confidence-name">YOLO Det.</span>
                      <div className="confidence-bar-bg">
                        <div
                          className="confidence-bar-fill"
                          style={{ width: `${v.detection_confidence}%` }}
                        />
                      </div>
                      <span className="confidence-pct">{v.detection_confidence}%</span>
                    </div>

                    <div className="confidence-row">
                      <span className="confidence-name">Classifier</span>
                      <div className="confidence-bar-bg">
                        <div
                          className="confidence-bar-fill"
                          style={{ width: `${v.classification_confidence}%` }}
                        />
                      </div>
                      <span className="confidence-pct">{v.classification_confidence}%</span>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
