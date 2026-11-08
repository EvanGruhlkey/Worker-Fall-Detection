import type { CameraInfo, Location } from "../types";
import CameraControls from "./CameraControls";
import Icon from "./Icon";

/** Live feed from the backend, with pose skeletons, track IDs and fall alerts drawn on. */
export default function CameraFeed({ camera, location, onChanged }: { camera: CameraInfo; location: Location; onChanged: () => void }) {
  const live = camera.online && camera.stream_url;
  return (
    <article className="camera">
      <div className="video">
        {live ? (
          // MJPEG stream; keyed by URL so a new camera session reconnects.
          <img key={camera.stream_url} className="stream" src={camera.stream_url!} alt={`${camera.name} live feed`} />
        ) : (
          <div className="placeholder">
            <Icon name="cam" size={40} className="big" />
            <span>Camera offline</span>
          </div>
        )}
        <span className="badge live-badge"><i className="dot" />LOCAL CAM</span>
        <span className="badge fps">{live ? `LIVE · ${camera.fps} FPS` : "OFFLINE"}</span>
        <span className="badge loc-badge"><Icon name="pin" />{location.name}</span>
      </div>
      <div className="cam-info">
        <div>
          <strong>{camera.name}</strong>
          <small>{camera.id} · {camera.label}</small>
        </div>
        {camera.online && <span className="online"><Icon name="wifi" />Online</span>}
      </div>
      <CameraControls camera={camera} onChanged={onChanged} />
    </article>
  );
}
