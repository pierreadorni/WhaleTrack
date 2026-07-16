import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { CircleMarker, MapContainer, Marker, Polygon, Polyline, TileLayer, useMap } from "react-leaflet";
import { divIcon } from "leaflet";
import { ChevronLeft, ChevronRight, Download, FileImage, Pause, Play, RotateCcw, ScanLine } from "lucide-react";
import "leaflet/dist/leaflet.css";
import "./styles.css";

const TRACK_COLORS = ["#18b6a4", "#f2b84b", "#f26d5b", "#5ac8ee", "#d79bff", "#a6d45d"];

function trackColor(trackKey, tracks) {
  const trackIndex = tracks.findIndex((track) => track.key === trackKey);
  return TRACK_COLORS[Math.max(trackIndex, 0) % TRACK_COLORS.length];
}

const BOAT_ICON = divIcon({ className: "boat-map-icon", html: "&#9973;", iconSize: [28, 28], iconAnchor: [14, 14] });

function api(path, options) {
  return fetch(path, options).then(async (response) => {
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  });
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function TrackBounds({ tracks }) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    const points = tracks.flatMap((track) => track.positions.map((position) => [position.latitude, position.longitude]));
    if (!fitted.current && points.length > 1) {
      map.fitBounds(points, { padding: [42, 42], maxZoom: 19 });
      fitted.current = true;
    }
  }, [map, tracks]);
  return null;
}

function MapPanel({ tracks, frameState }) {
  return (
    <MapContainer className="map" center={[22.8915, -109.8426]} zoom={17} zoomControl={false}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <TrackBounds tracks={tracks} />
      {frameState?.fov && <Polygon pathOptions={{ color: "#fbca57", weight: 2, fillColor: "#fbca57", fillOpacity: 0.14 }} positions={frameState.fov} />}
      {tracks.map((track, index) => {
        const color = trackColor(track.key, tracks);
        const segments = [[]];
        track.positions.forEach((position) => {
          const previous = segments.at(-1).at(-1);
          if (previous && position.time_s - previous.time_s > 0.5) segments.push([]);
          segments.at(-1).push(position);
        });
        return (
          <span key={track.key}>
            {segments.map((segment, segmentIndex) => (
              <Polyline key={segmentIndex} pathOptions={{ color, weight: 3, opacity: 0.8 }} positions={segment.map((position) => [position.latitude, position.longitude])} />
            ))}
          </span>
        );
      })}
      {frameState?.positions?.map((position) => {
        const trackKey = `${position.class_id}:${position.track_id}`;
        const color = trackColor(trackKey, tracks);
        if (position.class_name === "boat") return <Marker icon={BOAT_ICON} key={position.sequence} position={[position.latitude, position.longitude]} />;
        return <CircleMarker center={[position.latitude, position.longitude]} key={position.sequence} pathOptions={{ color: "#f9fbf6", weight: 2, fillColor: color, fillOpacity: 1 }} radius={8} />;
      })}
    </MapContainer>
  );
}

function VideoOverlay({ frameState, metadata, tracks }) {
  if (!metadata || !frameState?.positions?.length) return null;
  return (
    <svg className="video-overlay" viewBox={`0 0 ${metadata.width} ${metadata.height}`} preserveAspectRatio="none">
      {frameState.positions.map((position) => (
        <g key={position.sequence}>
          <polygon
            className="obb"
            fill={trackColor(`${position.class_id}:${position.track_id}`, tracks)}
            fillOpacity="0.14"
            points={position.corners.map(([x, y]) => `${x * metadata.width},${y * metadata.height}`).join(" ")}
            stroke={trackColor(`${position.class_id}:${position.track_id}`, tracks)}
          />
          <text className="track-label" fill={trackColor(`${position.class_id}:${position.track_id}`, tracks)} x={position.pixel[0] + 12} y={position.pixel[1] - 12}>{position.class_name === "boat" ? `${position.estimated_length_m?.toFixed(1)} m` : `ID ${position.track_id}`}</text>
        </g>
      ))}
    </svg>
  );
}

function BoatMetrics({ metrics }) {
  const summary = metrics?.summary;
  const samples = metrics?.samples || [];
  if (!summary || !summary.count) return <aside className="metrics-panel metrics-empty"><span>Boat validation: awaiting boat detections</span><a className="icon-button csv-export" href="/api/boat-metrics.csv" title="Download boat validation CSV" aria-label="Download boat validation CSV"><Download size={16} /></a></aside>;
  const plots = [
    ["altitude_m", "Altitude (m)"],
    ["pitch_deg", "Pitch (deg)"],
    ["distance_from_image_center_px", "Distance from image center (px)"],
  ];
  return <aside className="metrics-panel">
    <div className="metrics-heading"><strong>Boat validation</strong><span>reference {metrics.true_length_m} m · n={summary.count}</span><a className="icon-button csv-export" href="/api/boat-metrics.csv" title="Download boat validation CSV" aria-label="Download boat validation CSV"><Download size={16} /></a></div>
    <div className="metrics-values">
      {[['median_error_m', 'Median error'], ['mean_error_m', 'Mean error'], ['rmse_m', 'RMSE'], ['signed_bias_m', 'Signed bias'], ['median_absolute_error_m', 'Median absolute error'], ['mean_absolute_error_m', 'MAE']].map(([key, label]) => <span key={key}><b>{summary[key].toFixed(2)} m</b>{label}</span>)}
    </div>
    <div className="metric-plots">{plots.map(([key, label]) => {
      const values = samples.map((sample) => sample[key]);
      const min = Math.min(...values); const max = Math.max(...values) || min + 1;
      return <div className="metric-plot" key={key}><label>{label}</label><svg viewBox="0 0 180 54" role="img" aria-label={`${label} scatterplot`}>
        {samples.map((sample) => <circle key={sample.sequence} cx={10 + ((sample[key] - min) / (max - min || 1)) * 160} cy={50 - Math.min(1, Math.max(0, (sample.estimated_length_m - 4) / 8)) * 40} r="2.2" />)}
        <line x1="0" x2="180" y1={50 - Math.min(1, Math.max(0, (metrics.true_length_m - 4) / 8)) * 40} y2={50 - Math.min(1, Math.max(0, (metrics.true_length_m - 4) / 8)) * 40} />
      </svg></div>;
    })}</div>
  </aside>;
}

function App() {
  const query = new URLSearchParams(globalThis.location.search);
  const videoRef = useRef(null);
  const cursorRef = useRef(0);
  const loadingPositionsRef = useRef(false);
  const requestedTimeRef = useRef(Number.parseFloat(query.get("time")));
  const [session, setSession] = useState(null);
  const [positions, setPositions] = useState([]);
  const [frameState, setFrameState] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState("");
  const [paperMode, setPaperMode] = useState(query.get("figure") === "1");
  const [boatMetrics, setBoatMetrics] = useState(null);

  async function refresh() {
    try {
      const nextSession = await api("/api/session");
      setSession(nextSession);
      api("/api/boat-metrics").then(setBoatMetrics).catch(() => {});
      if (!loadingPositionsRef.current) {
        loadingPositionsRef.current = true;
        const update = await api(`/api/positions?after=${cursorRef.current}`);
        cursorRef.current = update.cursor;
        if (update.positions.length) setPositions((previous) => [...previous, ...update.positions]);
        loadingPositionsRef.current = false;
      }
      setError("");
    } catch (requestError) {
      loadingPositionsRef.current = false;
      setError(requestError.message || "Unable to reach the analysis API.");
    }
  }

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 900);
    return () => window.clearInterval(timer);
  }, []);

  const fps = session?.fps || 30;
  const totalFrames = session?.total_frames || 0;
  const duration = totalFrames / fps;
  const processedFrames = session?.inference?.processed_frames || 0;
  const analysisEnd = session?.inference?.done ? duration : Math.max(0, processedFrames / fps);
  const currentFrame = Math.min(Math.floor(currentTime * fps), Math.max(0, totalFrames - 1));

  useEffect(() => {
    const requestedTime = requestedTimeRef.current;
    if (!Number.isFinite(requestedTime) || requestedTime < 0 || analysisEnd < requestedTime || !videoRef.current) return;
    seek(requestedTime);
    requestedTimeRef.current = null;
  }, [analysisEnd]);

  useEffect(() => {
    if (!session || currentFrame >= processedFrames && !session.inference.done) {
      setFrameState(null);
      return;
    }
    let stale = false;
    api(`/api/frame-state?frame=${currentFrame}`).then((state) => {
      if (!stale) setFrameState(state);
    }).catch(() => {});
    return () => { stale = true; };
  }, [currentFrame, processedFrames, session?.inference?.done]);

  const tracksById = new Map();
  positions.forEach((position) => {
    const trackKey = `${position.class_id}:${position.track_id}`;
    if (!tracksById.has(trackKey)) tracksById.set(trackKey, []);
    tracksById.get(trackKey).push(position);
  });
  const tracks = [...tracksById.entries()].map(([key, trackPositions]) => ({ key, id: trackPositions[0].track_id, className: trackPositions[0].class_name, positions: trackPositions }));
  const progress = totalFrames ? (processedFrames / totalFrames) * 100 : 0;

  function seek(time) {
    const nextTime = Math.min(Math.max(time, 0), analysisEnd);
    if (videoRef.current) videoRef.current.currentTime = nextTime;
    setCurrentTime(nextTime);
  }

  function togglePaperMode() {
    const nextPaperMode = !paperMode;
    const url = new URL(globalThis.location.href);
    if (nextPaperMode) {
      url.searchParams.set("figure", "1");
      url.searchParams.set("time", currentTime.toFixed(2));
    } else {
      url.searchParams.delete("figure");
      url.searchParams.delete("time");
    }
    globalThis.history.replaceState({}, "", url);
    setPaperMode(nextPaperMode);
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) video.play(); else video.pause();
  }

  function handleTimeUpdate() {
    const video = videoRef.current;
    if (!video) return;
    if (!session?.inference?.done && video.currentTime > analysisEnd) {
      video.pause();
      video.currentTime = analysisEnd;
    }
    setCurrentTime(video.currentTime);
  }

  async function restart() {
    if (!window.confirm("Restart the analysis and clear the cached positions?")) return;
    setPositions([]);
    cursorRef.current = 0;
    await api("/api/restart", { method: "POST" });
    refresh();
  }

  return (
    <main className={`app-shell ${paperMode ? "paper-mode" : ""}`}>
      <header className="topbar">
        <div className="brand"><ScanLine size={23} /> <span>Whale Track</span><small>drone geolocation console</small></div>
        {paperMode ? (
          <div className="paper-status">Frame {formatTime(currentTime)} <span /> {tracks.length} stable tracks</div>
        ) : (
          <div className="run-status">
            <span className={`status-dot ${session?.inference?.running ? "live" : ""}`} />
            <span>{session?.inference?.done ? "Analysis complete" : session?.inference?.running ? "Inference running" : "Waiting"}</span>
            <strong>{Math.round(progress)}%</strong>
          </div>
        )}
        <button className="icon-button figure-mode-toggle" onClick={togglePaperMode} title="Toggle paper figure mode" aria-label="Toggle paper figure mode"><FileImage size={18} /></button>
      </header>

      <section className="workspace">
        <section className="video-stage" aria-label="Drone video preview">
          <div className="panel-head"><span>{paperMode ? "a. Video observations" : "Live video"}</span><span>{formatTime(currentTime)} / {formatTime(duration || 0)}</span></div>
          <div className="media-wrap">
            <video
              ref={videoRef}
              src="/api/video"
              muted
              playsInline
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={() => setCurrentTime(0)}
            />
            <VideoOverlay frameState={frameState} metadata={session} tracks={tracks} />
            {!session?.inference?.done && currentTime >= analysisEnd && <div className="analysis-frontier">Awaiting analysed frames</div>}
          </div>
          <div className="transport">
            <button className="icon-button" onClick={() => seek(currentTime - 1 / fps)} title="Previous frame" aria-label="Previous frame"><ChevronLeft size={19} /></button>
            <button className="play-button" onClick={togglePlayback} title={playing ? "Pause" : "Play"} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={21} fill="currentColor" /> : <Play size={21} fill="currentColor" />}</button>
            <button className="icon-button" onClick={() => seek(currentTime + 1 / fps)} title="Next frame" aria-label="Next frame"><ChevronRight size={19} /></button>
            <label className="speed-control">Speed
              <select value={speed} onChange={(event) => { const next = Number(event.target.value); setSpeed(next); videoRef.current.playbackRate = next; }}>
                {[0.25, 0.5, 1, 1.5, 2].map((value) => <option value={value} key={value}>{value}x</option>)}
              </select>
            </label>
            <button className="icon-button restart" onClick={restart} disabled={session?.inference?.running} title="Restart analysis" aria-label="Restart analysis"><RotateCcw size={18} /></button>
          </div>
        </section>

        <section className="map-stage" aria-label="Geographic whale tracks">
          <div className="panel-head"><span>{paperMode ? "b. Georeferenced trajectories" : "Geographic tracks"}</span><span>{tracks.length} stable IDs</span></div>
          <MapPanel tracks={tracks} frameState={frameState} />
          <div className="map-key"><i className="fov-key" /> Drone field of view <i className="track-key" /> Smoothed track</div>
          {frameState?.coordinates_paused && <div className="rotation-note">Coordinates paused during gimbal rotation ({frameState.yaw_rate_deg_s.toFixed(1)} deg/s)</div>}
        </section>
      </section>

      <section className="timeline-area" aria-label="Analysis progress and video seeking">
        <div className="timeline-meta"><span>Processed frames <b>{processedFrames.toLocaleString()} / {totalFrames.toLocaleString()}</b></span><span>{session?.position_count?.toLocaleString() || 0} reprojected positions</span></div>
        <div className="analysis-progress" aria-label="Inference progress"><span style={{ width: `${progress}%` }} /></div>
        <input className="seek" type="range" min="0" max={Math.max(analysisEnd, 0.01)} step="0.01" value={Math.min(currentTime, analysisEnd)} onChange={(event) => seek(Number(event.target.value))} disabled={!processedFrames} />
        <div className="timeline-labels"><span>0:00</span><span>{formatTime(analysisEnd)} analysed</span><span>{formatTime(duration || 0)}</span></div>
      </section>
      <BoatMetrics metrics={boatMetrics} />
      {error && <div className="error-banner">{error}</div>}
    </main>
  );
}

export default App;

const root = globalThis.__whaleTrackRoot ?? createRoot(document.getElementById("root"));
globalThis.__whaleTrackRoot = root;
root.render(<App />);