from typing import Dict

import cv2
import os
from typing import List
import numpy as np
from math import fabs


class DataWriter:

    def __init__(self, destination: str,
                 width: int,
                 height: int):
        self.destination: str = destination
        self.width: int = width
        self.height: int = height

    def write_to_image(self, filename: str, image: np.ndarray, img_format: str, extension=None) -> None:
        image: np.ndarray = self.make_image_size(image)
        image_file: str = self.destination

        if img_format.upper() == "JPEG":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not os.path.exists(image_file):
            os.makedirs(image_file)
        if extension:
            image_file: str = os.path.join(image_file, "%s.%s" % (str(filename), str(extension)))
        else:
            image_file: str = os.path.join(image_file, "%s.%s" % (str(filename), str(img_format)))
        cv2.imwrite(image_file, image)

    def make_image_size(self, img: np.ndarray) -> np.ndarray:
        img_height: int = img.shape[0]
        img_width: int = img.shape[1]
        if img_height == self.height and img_width == self.width:
            return img

        scale_x: float = self.width / img.shape[1]
        scale_y: float = self.height / img.shape[0]
        img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
        return img
