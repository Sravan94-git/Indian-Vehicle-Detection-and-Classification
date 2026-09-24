import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

function App() {
  const [backendState, setBackendState] = useState("checking");
  const [retryKey, setRetryKey] = useState(0);
  const [mode, setMode] = useState("image");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const accept = mode === "image" ? "image/png,image/jpeg" : "video/mp4,video/quicktime,video/x-msvideo";

  useEffect(() => {
    let cancelled = false;
    const deadline = Date.now() + 150000;

    async function checkBackend() {
      while (!cancelled && Date.now() < deadline) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        try {
          const response = await fetch(`${API_URL}/health`, { signal: controller.signal, cache: "no-store" });
          const payload = await response.json();
          if (payload.status === "ok") {
            if (!cancelled) setBackendState("ready");
            return;
          }
          if (payload.status === "error") throw new Error(payload.detail || "The backend could not load its models.");
        } catch {
          // Render can take several requests to wake the service.
        } finally {
          clearTimeout(timeout);
        }
        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
      if (!cancelled) setBackendState("error");
    }

    setBackendState("checking");
    checkBackend();
    return () => { cancelled = true; };
  }, [retryKey]);

  if (backendState !== "ready") {
    return <BackendLoading state={backendState} onRetry={() => setRetryKey((key) => key + 1)} />;
  }

  function chooseMode(nextMode) {
    setMode(nextMode);
    setFile(null);
    setResult(null);
    setError("");
  }

  async function analyze(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose a file before starting the analysis.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_URL}/api/analyze/${mode}`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The analysis failed.");
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || "Could not reach the Vehix API.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <nav><div className="brand"><span className="brand-mark" />Vehix</div><div className="nav-note">Vehicle intelligence / 01</div></nav>
      <section className="hero">
        <div><div className="eyebrow">Indian road vision system</div><h1>See the road<br /><em>in detail.</em></h1><p className="hero-copy">Turn traffic imagery into a readable vehicle report. Vehix detects vehicles first, then identifies their type with a dedicated classifier.</p></div>
        <form className="upload-panel" onSubmit={analyze}>
          <div className="panel-kicker">Start an analysis</div>
          <div className="mode-switch"><button type="button" className={mode === "image" ? "mode-btn active" : "mode-btn"} onClick={() => chooseMode("image")}>Image</button><button type="button" className={mode === "video" ? "mode-btn active" : "mode-btn"} onClick={() => chooseMode("video")}>Video</button></div>
          <label className="dropzone" htmlFor="file"><span className="upload-icon">↥</span><strong>{file ? file.name : `Choose a ${mode} to inspect`}</strong><span>{mode === "image" ? "PNG or JPEG · annotated result" : "MP4, AVI or MOV · processed frame by frame"}</span><input id="file" type="file" accept={accept} onChange={(event) => { setFile(event.target.files?.[0] || null); setResult(null); }} /></label>
          {error && <p className="error">{error}</p>}
          <button className="submit-btn" disabled={busy}>{busy ? "Analyzing..." : "Run detection →"}</button>
        </form>
      </section>
      {result && <Result result={result} />}
      <section className="methodology"><div className="section-heading"><div><div className="eyebrow">Methodology</div><h2>From pixels to insight.</h2></div><p>A two-stage pipeline keeps detection broad and classification specific.</p></div><div className="steps"><Step number="01 / INPUT" title="Capture" text="An image or video frame enters the analysis pipeline." /><Step number="02 / DETECT" title="Locate" text="YOLOv8 scans the scene and locates every vehicle." /><Step number="03 / CLASSIFY" title="Identify" text="Each crop is passed to the trained vehicle classifier." /><Step number="04 / REPORT" title="Understand" text="Confidence scores and annotated media become a report." /></div></section>
    </main>
  );
}

function BackendLoading({ state, onRetry }) {
  const failed = state === "error";
  return <main className="loading-screen"><div className="loading-card"><span className="brand-mark" /><div className="eyebrow">Vehix / startup check</div><h1>{failed ? "The API needs another try." : "Waking the vehicle lab."}</h1><p>{failed ? "The backend did not respond within two minutes. Render may still be starting the service." : "The analysis engine is starting on Render. This usually takes up to a minute on the first visit."}</p>{failed ? <button className="submit-btn" onClick={onRetry}>Try again →</button> : <div className="loading-indicator" aria-label="Waiting for the Vehix backend"><span /><span /><span /></div>}</div></main>;
}

function Step({ number, title, text }) { return <article className="step"><div className="step-number">{number}</div><h3>{title}</h3><p>{text}</p></article>; }

function Result({ result }) {
  if (result.type === "video") return <section className="report"><div className="eyebrow">Analysis report / video</div><h2>Motion, made readable.</h2><video controls src={result.result_url} /><div className="stats"><Stat value={result.processed_frames} label="frames with detections" /><Stat value={result.fps} label="effective FPS" /><Stat value={`${result.elapsed_time}s`} label="processing time" /></div></section>;
  return <section className="report"><div className="eyebrow">Analysis report / image</div><h2>What the camera found.</h2><div className="result-grid"><img src={result.result_url} alt="Annotated vehicle scene" /><div><div className="stat"><strong>{result.vehicles.length}</strong><span>vehicles found</span></div>{result.vehicles.map((vehicle) => <article className="vehicle" key={`${vehicle.number}-${vehicle.crop_url}`}><img src={vehicle.crop_url} alt={`Vehicle ${vehicle.number}`} /><div><strong>{vehicle.label}</strong><span>Detection {vehicle.detection_confidence}% · Classification {vehicle.classification_confidence}%</span></div></article>)}</div></div></section>;
}

function Stat({ value, label }) { return <div className="stat"><strong>{value}</strong><span>{label}</span></div>; }

createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
