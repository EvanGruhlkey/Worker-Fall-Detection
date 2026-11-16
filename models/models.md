# Models

The default pose pipeline uses these files:

- `pose_pipeline.json`: selects the detector and pose model. The default uses YOLO26m.
- `yolo26m.onnx`: person detector (default). Finds far more people mid-fall than YOLO26n
  (74% vs 45% of people kept during hard falls), but needs a GPU to run live.
- `rtmpose-t.onnx`: 17-keypoint pose estimator.
- `pose_pipeline_cpu.json` + `yolo26n.onnx`: the smaller detector, for laptops without a GPU.
  Less accurate, and the fall model is trained on YOLO26m poses, so use it only when
  YOLO26m is too slow.

`yolo26n.onnx` and `rtmpose-t.onnx` were copied from the existing `comparison/` models. The RTMPose
source is recorded in `comparison/preparation.json`. `yolo26m.onnx` was exported from the
Ultralytics `yolo26m.pt` the same way (640 px, opset 17, no built-in NMS):

```python
YOLO("yolo26m.pt").export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, end2end=False)
```

The pipeline selects CUDA, DirectML, or CPU through ONNX Runtime. See `src/src.md` for setup.

## Speed (live pipeline, 12 frames per second needed)

Measured 2026-11-15 on a laptop with an NVIDIA T600 (4 GB) through DirectML:

| Detector | 2 people in view | 8 people | ~10 people |
|---|---:|---:|---:|
| YOLO26m (default) | 18 fps | 12 fps | 11 fps |
| YOLO26n (`pose_pipeline_cpu.json`) | | | 14.5 fps |

Without a GPU (ONNX on the CPU) YOLO26m runs at about 1 fps and YOLO26n at about 3 fps, too slow
for live; recorded videos still process, just slowly.

## Choosing the config

The dashboard uses `pose_pipeline.json`. On a laptop without a GPU, set this in `.env`:

```
FALL_MODEL_PATH=models/pose_pipeline_cpu.json
```

To force the CPU even on a laptop with a GPU (to see how a laptop without one performs), set
`FALL_DEVICE=cpu` in `.env`; remove it (or set `auto`) to use the GPU again.

The `.pt` files are retained for training and the separate temporal fall classifier. The pose pipeline does not load them.
