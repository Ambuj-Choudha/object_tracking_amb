## YOLO v10

This repo contains YOLOv10 wrapper using the standard pytorch model and the ONNX model.


[model link for pretrained YOLOv10-m](https://huggingface.co/jameslahm/yolov10m)
[[ONNX model link for YOLOv10-m]](https://huggingface.co/onnx-community/yolov10m/tree/main)

### Usage

`bash
uv venv --python 3.9 venv
`

`bash
uv pip install -r requirements.txt
`

`bash
git clone https://github.com/THU-MIG/yolov10.git
`

Install the repo as an editable package (basically add a pointer to the repo, in the local environment's site-packages folder)
`bash
cd yolov10
`

`bash
pip install -e .
`

Asumming you are in YOLOv10\ directory:

`bash
python src\main.py
`


### Performance

**Speed from .pt model**:
Pre-processing time : 12.008 ms
Inference time : 2283.040 ms
Post-processing time : 33.775 ms

**Speed from ONNX model**:
Pre-processing time : 14.318 ms
Inference time : 1357.630 ms
Post-processing time : 2.294 ms


### References

I followed this [tutorial](https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html) from openCV