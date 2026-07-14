import os
import onnxruntime as ort
from ..object_tracker import config

def main():
    session = ort.InferenceSession(config.ONNX_MODEL)

    for inp in session.get_inputs():
        print(inp.name, inp.shape, inp.type)
    # e.g. → images  [1, 3, 640, 640]  tensor(float)

    for out in session.get_outputs():
        print(out.name, out.shape, out.type)
    # e.g. → output0  [1, 300, 6]  tensor(float)


if __name__ == "__main__":
    main()
