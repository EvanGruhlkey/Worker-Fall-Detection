# Models

The default pose pipeline uses these files:

- `pose_pipeline.json`: selects the detector and pose model.
- `yolo26n.onnx`: person detector.
- `rtmpose-t.onnx`: 17-keypoint pose estimator.

These ONNX files were copied from the existing `comparison/` models. The RTMPose source is recorded in `comparison/preparation.json`.

The pipeline selects CUDA, DirectML, or CPU through ONNX Runtime. See `src/src.md` for setup.

The `.pt` files are retained for training and the separate temporal fall classifier. The pose pipeline does not load them.
