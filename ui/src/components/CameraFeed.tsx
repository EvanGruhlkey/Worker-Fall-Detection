import { LOCATION } from "../data/mock";
import type { CameraInfo } from "../types";
import Icon from "./Icon";

/** Placeholder only: no webcam access yet. Swap with the YOLO model withbounding boxes */
export default function CameraFeed({ camera }: { camera: CameraInfo }) {
  return (
    <article className="camera">
      <div className="video">
        <span className="badge live-badge"><i className="dot" />LOCAL CAM</span>
        <span className="badge fps">LIVE · {camera.fps} FPS</span>
        <div className="placeholder">
          <Icon name="cam" size={40} className="big" />
          <span>Camera preview placeholder</span>
        </div>
        <span className="badge loc-badge"><Icon name="pin" />{LOCATION.name}</span>
      </div>
      <div className="cam-info">
        <div>
          <strong>{camera.name}</strong>
          <small>{camera.id} · {camera.label}</small>
        </div>
        {camera.online && <span className="online"><Icon name="wifi" />Online</span>}
      </div>
    </article>
  );
}
