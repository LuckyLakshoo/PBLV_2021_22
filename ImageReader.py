from typing import List

import cv2
import os
import numpy as np


class ImageReader:
    def __init__(self, input_path: str):
        self.input_path: str = input_path

    def get_files(self) -> List[str]:
        files: List[str] = []
        for file in os.listdir(self.input_path):
            if file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".jpg") or file.endswith(".png") or file.endswith(".pgm"):
                files.append(file)
        return files

    def load_image(self, file: str) -> np.ndarray:
        image_path: str = os.path.join(self.input_path, file)
        img: np.ndarray = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_image_no_convert(self, file: str) -> np.ndarray:
        image_path: str = os.path.join(self.input_path, file)
        img: np.ndarray = cv2.imread(image_path)
        return img