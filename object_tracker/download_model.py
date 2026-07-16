"""Download the YOLOv10-m ONNX model from Hugging Face."""
import argparse
import logging
import os
import sys
import urllib.request
from urllib.error import URLError

from object_tracker import config

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://huggingface.co/onnx-community/yolov10m/resolve/main/onnx/model.onnx"


def download(url: str, dest: str, force: bool = False) -> None:
    if os.path.exists(dest) and not force:
        logger.info("Model already exists at %s (use --force to overwrite)", dest)
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)

    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MiB
            while True:
                buf = response.read(chunk)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded * 100 / total
                    print(
                        f"\r  {downloaded / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
        os.replace(tmp, dest)
        logger.info("Saved to %s", dest)
    except (URLError, OSError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        logger.error("Download failed: %s", e)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLOv10-m ONNX model")
    parser.add_argument("--url", default=DEFAULT_URL, help="Model download URL")
    parser.add_argument("--dest", default=config.ONNX_MODEL, help="Destination path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    download(args.url, args.dest, args.force)


if __name__ == "__main__":
    main()
