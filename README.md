## YOLO v10

This repo contains YOLOv10 wrapper using ONNX model under the hood.

[ONNX model for YOLOv10-m](https://huggingface.co/onnx-community/yolov10m/tree/main)

### Setup

Requirements: [uv](https://docs.astral.sh/uv/) · Python 3.10+

```bash
uv sync
```

This installs all dependencies from `uv.lock` and creates a `.venv` automatically.

### Usage

After `uv sync`, a `detect` command is available in the environment:

```bash
uv run detect --input-image horse.jpg
```

Or activate the virtual environment and run directly:

```bash
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

detect --input-image horse.jpg
```

By default the command reads from the configured input folder and writes results next to the input file with a `_detections` suffix.

### Development

Install dev dependencies (pytest, mypy, ruff):

```bash
uv sync --extra dev
```

Run tests:

```bash
uv run pytest
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
