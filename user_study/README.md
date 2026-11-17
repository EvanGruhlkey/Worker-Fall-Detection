# User study

Participants watch a 3×3 camera wall for two minutes, then solve arithmetic problems while automated alerts monitor the cameras for two minutes. A short survey follows.

## 1. Get ready

Install **Git**, **Node.js**, and **uv**. Run these commands in PowerShell:

```powershell
git clone --branch user-study https://github.com/EvanGruhlkey/Worker-Fall-Detection.git
cd Worker-Fall-Detection
uv sync
npm.cmd ci --prefix user_study/frontend
npm.cmd run build --prefix user_study/frontend
```

Already have the repository? Switch to `user-study`, pull the latest changes, then run the install and build commands.

## 2. Add the videos

Ask the study team for the prepared footage. Copy it into:

```text
data/study/rig-wall-two/
  nine-manifest.json
  media/
    ...video files...
```

Videos are not included in Git. Keep their filenames unchanged. For a new collection, do not copy someone else's `study.sqlite3` database.

## 3. Start the app

From the repository root:

```powershell
$env:FALL_STUDY_DIR = "$PWD\data\study\rig-wall-two"
uv run uvicorn user_study.backend.app:app --host 127.0.0.1 --port 8765
```

Open [localhost:8765](http://localhost:8765/). Keep the terminal open. Press **Ctrl+C** to stop the server.

## 4. Run the study

1. Open **Study setup → Settings**.
2. Choose `nine-manifest.json` under **Replace study footage**, then **Save footage**. Do this once for a new database.
3. Select **Back to study**. The participant enters their background information and completes consent.
4. Complete manual monitoring, automated monitoring, and the survey.
5. Select **New participant** for the next person.

IDs are assigned automatically. No facilitator key is needed. Download responses from **Study setup → Results**.

## Common fixes

- **Missing videos:** check the `media/` folder. Uploading a manifest does not copy videos.
- **Port already in use:** stop the previous server before starting another.
- **Old page:** rebuild the frontend and refresh the browser.
- **Continue later:** restart with the same study folder and keep its database. Use the same browser and `localhost` address to recover the participant session.

## Development

With the backend running, use `npm.cmd run dev --prefix user_study/frontend` for live frontend updates.

For footage generation, model details, and test commands, see [Footage and validation](docs/footage-and-validation.md).
