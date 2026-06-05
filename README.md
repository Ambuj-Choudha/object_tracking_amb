## YOLO v10

This repo contains YOLOv10 wrapper using ONNX model under the hood.

[ONNX model for YOLOv10-m](https://huggingface.co/onnx-community/yolov10m/tree/main)

### Setup

Requirements: [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

This installs all dependencies from `uv.lock` and creates a `.venv` automatically.

### Usage

Assuming you are in the `object_tracker_amb/` directory:

```bash
uv run python -m scripts.main --input-image horse.jpg
```

Or activate the virtual environment and run directly:

```bash
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

python -m scripts.main --input-image horse.jpg
```

### Model Output
#### Input Image
<img src="docs/horse.jpg" alt="Input Image" width="600">

#### Detection Result
<img src="docs/horse_detections.jpg" alt="Detection result" width="600">

### Performance

- Pre-processing time : 14.318 ms
- Inference time : 1357.630 ms
- Post-processing time : 2.294 ms

### References

This [tutorial](https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html) from openCV is quite helpful
