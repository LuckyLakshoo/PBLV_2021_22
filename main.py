from ImageReader import ImageReader
from BillboardService import BillboardService
from DataWriter import DataWriter
import os
import numpy as np
from typing import List
from tqdm import tqdm

def create_images_list(files):
    images = []
    for file in files:
        images.append(file)
    return images


if __name__ == '__main__':
    bg_reader: ImageReader = ImageReader("./data/bg")
    # bg_id: int = randint(0, len() - 1)
    bg_path_nat = os.path.abspath("./data/bg/bg_nat_2.png")
    bg_path_white = os.path.abspath("./data/bg/bg_white.png")
    bg_img_nat: np.ndarray = bg_reader.load_image(bg_path_nat)
    bg_img_shape_nat = bg_img_nat.shape
    bg_img_white: np.ndarray = bg_reader.load_image(bg_path_white)
    bg_img_shape_white = bg_img_white.shape

    images: List[str] = os.listdir("./data/nat")
    masks: List[str] = os.listdir("./data/masks")
    files = []
    input_folder_images = os.path.abspath("./data/nat")
    input_folder_masks = os.path.abspath("./data/masks")
    for index in range(len(images)):
        files.append((os.path.join(input_folder_images, images[index]), os.path.join(input_folder_masks, masks[index])))
    billboard_service = BillboardService(bg_img_shape_nat[1], bg_img_shape_nat[0])
    results = []
    for index in tqdm(range(2)):
        results.append(billboard_service.add_multiple_background(
            img_files=files,
            bg_img_nat=bg_img_nat.copy(),
            bg_img_white=bg_img_white.copy()
        ))
    writer_images: DataWriter = DataWriter(os.path.abspath("./data/out/images"), width=bg_img_shape_nat[1], height=bg_img_shape_nat[0])
    writer_masks: DataWriter = DataWriter(os.path.abspath("./data/out/annotations"), width=bg_img_shape_white[1], height=bg_img_shape_white[0])
    index: int = 0
    for result in results:
        for res in result:
            writer_images.write_to_image(f"{index}", res[0], "png")
            writer_masks.write_to_image(f"{index}", res[1], "png")
            index += 1
