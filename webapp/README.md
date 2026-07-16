# Whale Track Web App

This app presents a browser video player alongside a Leaflet map of the whale tracks. Flask runs the tracker in the background and persists completed results in `webapp/data/`; React only lets the video play through frames that have already been analysed.

## Run locally

In one terminal, start the API and background inference using the existing project environment:

```bash
source venv/bin/activate
python webapp/server.py --video video.mp4 --srt metadata.srt --model model.pt --calibration camera_calibration_DJIM3T_RGBwide.json
```

In a second terminal, start the React client:

```bash
cd webapp/frontend
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`.

`POST /api/restart` from the restart control clears a completed session cache and reruns the analysis. The control is unavailable while an inference worker is active. An interrupted analysis starts fresh on the next launch because BOTSORT's in-memory state cannot be resumed safely. Change a tracking option on the Flask command line to create a separate cache configuration.