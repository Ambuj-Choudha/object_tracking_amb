## YOLO v10

This repo contains YOLOv10 wrapper using ONNX model under the hood.

[ONNX model for YOLOv10-m](https://huggingface.co/onnx-community/yolov10m/tree/main)

### Setup

Requirements: [uv](https://docs.astral.sh/uv/) · Python 3.10+

```bash
uv sync
```

This installs all dependencies from `uv.lock` and creates a `.venv` automatically.

### Download the model

After setting up the environment, to automatically download the model, just run the command below:

```bash
uv run download-model
```
This would download model from hugging face model repository.

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

By default the command reads from the configured input folder and writes results in the data/output folder with a `_detections` suffix.

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

### TODOs


- [ ] add batch processing for multiple images
- [ ] add functionality for video file as input
- [ ] add per-class confidence thresholds (and `--confidence` CLI override)
- [ ] add JSON export of detections (`--output-json`)
- [ ] support arbitrary input paths and `--input-dir` for folder input
- [ ] add model auto-download helper (avoids manual HF step)
- [ ] add a Tracker to the project

### References

This [tutorial](https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html) from openCV is quite helpful
