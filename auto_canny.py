"""
Runs canny edge-detection algorithm on supplied images, outputs to output/ dir

adapted from https://bit.ly/2HJtVc7
"""

import ntpath
import os
import argparse
import cv2
import numpy


def auto_canny(image: str, sigma: float = 0.33) -> numpy.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median = numpy.median(blurred)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edged = cv2.Canny(image, lower, upper)
    return edged


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "images",
        metavar="image",
        type=argparse.FileType("r"),
        nargs="+",
        help="image filenames to process, as positional args"
    )
    arg_parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        type=__dir_path,
        help="output, directory. where to write the edge detection images"
    )
    args = vars(arg_parser.parse_args())

    for image_path in [i.name for i in args["images"]]:
        auto = auto_canny(image=cv2.imread(image_path))
        filename = ntpath.split(image_path)[-1]
        cv2.imwrite(os.path.join(args["out_dir"], filename), auto)
        cv2.waitKey(0)


def __dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a directory or doesn't exist")


if __name__ == "__main__":
    main()
