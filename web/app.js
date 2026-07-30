// AerialDetector front-end: upload, poll progress, play result, download.
const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("file-input");
const runBtn = $("run-btn");
let selectedFile = null;
let jobId = null;
let pollTimer = null;

// ---- file selection (click + drag/drop) --------------------------------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

function setFile(f) {
  if (!f) return;
  if (!f.type.startsWith("video/")) {
    showError("That file isn't a video. Choose an MP4, MOV, AVI, or MKV file.");
    return;
  }
  selectedFile = f;
  $("file-name").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`;
  runBtn.disabled = false;
  hideError();
}

// ---- confidence slider label -------------------------------------------
$("opt-conf").addEventListener("input", (e) => {
  $("conf-val").textContent = Number(e.target.value).toFixed(2);
});

// ---- run ----------------------------------------------------------------
runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();

  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("model", $("opt-model").value);
  fd.append("conf", $("opt-conf").value);
  fd.append("slice_mode", $("opt-slice").checked);
  fd.append("track", $("opt-track").checked);

  show("progress-panel");
  hide("upload-panel");
  $("progress-text").textContent = "Uploading footage…";
  $("device-text").textContent = "";
  $("bar-fill").style.width = "0%";

  try {
    const res = await fetch("/api/detect", { method: "POST", body: fd });
    if (!res.ok) throw new Error("The upload was rejected by the server.");
    jobId = (await res.json()).job_id;
    pollTimer = setInterval(poll, 800);
  } catch (err) {
    fail(err.message);
  }
});

// ---- progress polling ---------------------------------------------------
async function poll() {
  try {
    const res = await fetch(`/api/status/${jobId}`);
    if (!res.ok) throw new Error("Lost contact with the processing job.");
    const s = await res.json();

    if (s.device) $("device-text").textContent = s.device;

    if (s.state === "queued" || s.state === "loading") {
      $("bar-fill").style.width = "6%";
      $("progress-text").textContent =
        s.state === "loading" ? "Loading model…" : "Queued…";
    } else if (s.state === "processing") {
      const pct = s.percent != null ? s.percent : null;
      $("bar-fill").style.width = pct != null ? `${pct}%` : "50%";
      const total = s.total ? ` of ${s.total}` : "";
      const fps = s.fps ? ` · ${s.fps} FPS` : "";
      const done = pct != null ? ` · ${pct}%` : "";
      $("progress-text").textContent = `Frame ${s.processed}${total}${fps}${done}`;
    } else if (s.state === "done") {
      clearInterval(pollTimer);
      showResult(s.summary);
    } else if (s.state === "error") {
      clearInterval(pollTimer);
      fail(s.error || "Processing did not complete.");
    }
  } catch (err) {
    clearInterval(pollTimer);
    fail(err.message);
  }
}

// ---- result -------------------------------------------------------------
function showResult(summary) {
  hide("progress-panel");
  show("result-panel");

  const v = $("result-video");
  v.src = `/api/result/${jobId}`; // streamed for playback, not saved to disk
  v.load();

  const totals = summary.class_totals || {};
  const totalDetections = Object.values(totals).reduce((a, b) => a + b, 0);
  const chips = Object.entries(totals)
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `<span class="chip">${k} · ${n}</span>`)
    .join("");

  const stats = [
    { value: summary.frames_processed, label: "frames processed" },
    { value: summary.avg_fps ?? "—", label: "frames per second" },
  ];
  if (summary.unique_tracks != null)
    stats.push({ value: summary.unique_tracks, label: "unique objects" });
  else
    stats.push({ value: totalDetections, label: "detections" });

  const statCards = stats
    .map((s) => `<div class="stat"><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`)
    .join("");

  $("summary").innerHTML = `
    <div class="stat-row">${statCards}</div>
    <div class="class-list">${chips || "<span class='field-hint'>No objects were detected in this clip.</span>"}</div>`;
}

$("download-btn").addEventListener("click", () => {
  window.location.href = `/api/result/${jobId}?download=true`;
});
$("json-btn").addEventListener("click", () => {
  window.location.href = `/api/detections/${jobId}`;
});
$("restart-btn").addEventListener("click", () => {
  if (jobId) fetch(`/api/job/${jobId}`, { method: "DELETE" }).catch(() => {});
  $("result-video").pause();
  $("result-video").src = "";
  jobId = null;
  selectedFile = null;
  fileInput.value = "";
  $("file-name").textContent = "";
  runBtn.disabled = true;
  hide("result-panel");
  show("upload-panel");
});

// ---- helpers ------------------------------------------------------------
function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }
function showError(msg) { const e = $("error"); e.textContent = msg; e.classList.remove("hidden"); }
function hideError() { $("error").classList.add("hidden"); }
function fail(msg) {
  showError(msg);
  hide("progress-panel");
  show("upload-panel");
}
