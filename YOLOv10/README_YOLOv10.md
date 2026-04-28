## YOLO v10

This repo contains YOLOv10 wrapper using ONNX model under the hood.

[ONNX model for YOLOv10-m](https://huggingface.co/onnx-community/yolov10m/tree/main)

### Usage

`
uv venv --python 3.9 venv
`

`
uv pip install -r requirements.txt
`

Asumming you are in YOLOv10\ directory:

`
python src\main.py --input-image horse.jpg
`

### Model Output
#### Input Image
<img src="../docs/horse.jpg" alt="Input Image" width="600">

#### Detection Result
<img src="../docs/horse_detections.jpg" alt="Detection result" width="600">

### Performance
 
- Pre-processing time : 14.318 ms <br>
- Inference time : 1357.630 ms <br>
- Post-processing time : 2.294 ms <br>


### References

This [tutorial](https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html) from openCV is quite helpful
