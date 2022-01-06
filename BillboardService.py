from typing import Tuple, List, TextIO

from BoundingBox import BoundingBox
import numpy as np
import cv2
import os
import random



class BillboardService:
    """
    Service for creating billboard and implanting the extracted object drawn onto the billboard onto a background
    """
    use_abs: bool = True
    DEFAULT_MAX_SCALE: float = 0.8
    DEFAULT_MIN_SCALE: float = 0.5

    def __init__(self, width, height):
        self.width: int = width
        self.height: int = height
        # left border
        self.X_MIN_IMPLANT_AREA: int = -int(width / 3)
        # right border
        self.X_MAX_IMPLANT_AREA: int = int(width / 3)
        # upper border
        self.Y_MIN_IMPLANT_AREA: int = -int(height / 3)
        # lower border
        self.Y_MAX_IMPLANT_AREA: int = int(height / 3)

    def random_move(self,
                     input_image: Tuple[np.ndarray, np.ndarray],
                     input_mask: Tuple[np.ndarray, np.ndarray],
                    m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Move the image random in the bg image.

        | Keyword arguments:
        | input_image -- input image to move
        | input_mask -- alpha mask of the image
        | m -- the move matrix

        | Output:
        | Tuple[np.ndarray, np.ndarray] : moved image and bg image
        """
        image: np.ndarray = np.copy(input_image[0])
        mask: np.ndarray = np.copy(input_mask[0])
        image_2: np.ndarray = np.copy(input_image[1])
        mask_2: np.ndarray = np.copy(input_mask[1])
        rows: int = image.shape[0]
        cols: int = image.shape[1]
        # move object
        image = cv2.warpAffine(image, m, (cols, rows))
        mask = cv2.warpAffine(mask, m, (cols, rows))
        image_2 = cv2.warpAffine(image_2, m, (cols, rows))
        mask_2 = cv2.warpAffine(mask_2, m, (cols, rows))
        return image, mask, image_2, mask_2


    def random_scale(self,
                     input_image: Tuple[np.ndarray, np.ndarray],
                     input_mask: Tuple[np.ndarray, np.ndarray],
                     move_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale the object depending on the y position of the bg, where it is going to get implanted onto

        | Keyword arguments:
        | input_image -- input image to scale
        | input_mask -- alpha mask of the input image
        | move_matrix -- move matrix is needed for y position to get the corresponding scale (higher -> smaller, lower -> bigger)

        | Output:
        | Tuple[np.ndarray, np.ndarray] -- rescaled image and mask
        """
        image: np.ndarray = np.copy(input_image[0])
        mask: np.ndarray = np.copy(input_mask[0])
        image_2: np.ndarray = np.copy(input_image[1])
        mask_2: np.ndarray = np.copy(input_mask[1])
        org_height: int = image.shape[0]
        org_width: int = image.shape[1]
        y_pos_object: np.float32 = move_matrix[1][2]
        max_scale = self.DEFAULT_MAX_SCALE
        min_scale = self.DEFAULT_MIN_SCALE
        diff_value_of_scale: float = (max_scale - min_scale)
        implant_area_y: int = self.Y_MAX_IMPLANT_AREA - self.Y_MIN_IMPLANT_AREA
        ran_scale: float = diff_value_of_scale * (
                (y_pos_object - self.Y_MIN_IMPLANT_AREA) / implant_area_y) + min_scale
        image: np.ndarray = cv2.resize(image, None, fx=ran_scale, fy=ran_scale, interpolation=cv2.INTER_CUBIC)
        image_2: np.ndarray = cv2.resize(image_2, None, fx=ran_scale, fy=ran_scale, interpolation=cv2.INTER_CUBIC)
        height: int = image.shape[0]
        width: int = image.shape[1]
        x_padding: int = (org_width - width) // 2
        y_padding: int = (org_height - height) // 2
        image = cv2.copyMakeBorder(
            image,
            top=y_padding,
            bottom=y_padding,
            left=x_padding,
            right=x_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        image_2 = cv2.copyMakeBorder(
            image_2,
            top=y_padding,
            bottom=y_padding,
            left=x_padding,
            right=x_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        image = cv2.resize(image, (org_width, org_height))
        image_2 = cv2.resize(image_2, (org_width, org_height))
        mask = cv2.resize(mask, None, fx=ran_scale, fy=ran_scale, interpolation=cv2.INTER_CUBIC)
        mask_2 = cv2.resize(mask_2, None, fx=ran_scale, fy=ran_scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.copyMakeBorder(
            mask,
            top=y_padding,
            bottom=y_padding,
            left=x_padding,
            right=x_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )
        mask_2 = cv2.copyMakeBorder(
            mask_2,
            top=y_padding,
            bottom=y_padding,
            left=x_padding,
            right=x_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )
        mask = cv2.resize(mask, (org_width, org_height))
        mask_2 = cv2.resize(mask_2, (org_width, org_height))

        if mask.sum() < 100:
            print("Error too small")
            return input_image[0], input_mask[1], input_image[1], input_mask[1]
        return image, mask, image_2, mask_2

    def laplacian_pyramid_blending_with_mask(self,
                                             A: np.ndarray,
                                             B: np.ndarray,
                                             M: np.ndarray,
                                             num_levels=4) -> np.ndarray:
        """
        Calculate the laplacian pyramid blending algorithm on the input image

        | Keyword arguments:
        | A -- input image
        | B -- bg image
        | M -- alpha mask of the input image
        | num_levels -- levels of laplacian algorithm

        | Output:
        | np.ndarray -- image with laplacian pyramid blending
        """

        # generate Gaussian pyramid for A,B and mask
        GA: np.ndarray = A.copy()
        GB: np.ndarray = B.copy()
        GM: np.ndarray = M.copy()
        gpA: List[np.ndarray] = [GA]
        gpB: List[np.ndarray] = [GB]
        gpM: List[np.ndarray] = [GM]

        for i in range(num_levels):
            GA = cv2.pyrDown(GA)
            GB = cv2.pyrDown(GB)
            GM = cv2.pyrDown(GM)
            gpA.append(np.float32(GA))
            gpB.append(np.float32(GB))
            gpM.append(np.float32(GM))

        # generate Laplacian Pyramids for A,B and masks
        lpA: List[np.ndarray] = [gpA[num_levels - 1]]
        lpB: List[np.ndarray] = [gpB[num_levels - 1]]
        gpMr: List[np.ndarray] = [gpM[num_levels - 1]]
        for i in range(num_levels - 1, 0, -1):
            LA: np.ndarray = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
            LB: np.ndarray = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
            lpA.append(LA)
            lpB.append(LB)
            gpMr.append(gpM[i - 1])  # reverse the masks

        # blend images according to mask in each level
        LS: List[np.ndarray] = []
        for la, lb, gm in zip(lpA, lpB, gpMr):
            ls = la * gm + lb * (1.0 - gm)
            LS.append(ls)

        # reconstruct
        ls_: np.ndarray = LS[0]
        for i in range(1, num_levels):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])
        ls_[ls_ < 0.0] = 0.0
        ls_[ls_ > 360.0] = 360.0

        return ls_

    def resize_object(self, matrix: np.ndarray, from_width: int, from_height: int, to_scale: float) -> np.ndarray:
        """
        Resize object draws white borders on top/left/right/bottom to get the correct scale.

        | Keyword arguments:
        | matrix -- input image
        | from_width -- input image width
        | from_height -- input image height
        | to_scale -- result scale

        | Output:
        | np.ndarray -- scaled image with white border but 0 alpha value
        """
        is_add_left_or_top: bool = False
        from_scale: float = from_width / from_height
        while from_scale != to_scale:
            if from_scale < to_scale:
                # change width
                from_width += 1
                if is_add_left_or_top:
                    b: np.ndarray = np.zeros((from_height, from_width, 4), dtype=np.uint8)
                    b[:, 1:from_width, :] = matrix  # input front
                else:
                    b: np.ndarray = np.zeros((from_height, from_width, 4), dtype=np.uint8)
                    b[:, :-1, :] = matrix  # input back
                is_add_left_or_top = not is_add_left_or_top
            else:
                # change height
                from_height += 1
                if is_add_left_or_top:
                    b: np.ndarray = np.zeros((from_height, from_width, 4), dtype=np.uint8)
                    b[1:from_height + 1, :, :] = matrix  # input front
                else:
                    b: np.ndarray = np.zeros((from_height, from_width, 4), dtype=np.uint8)
                    b[:-1, :, :] = matrix  # input back
                is_add_left_or_top = not is_add_left_or_top
            matrix: np.ndarray = b
            from_scale = from_width / from_height
        return matrix


    def get_border(self, alpha: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of the image

        | Keyword arguments:
        | alpha -- alpha mask of the image

        | Output:
        | Tuple[int, int, int, int] : xmin, ymin, xmax, ymax
        """
        opacity_layer: np.ndarray = alpha
        rows, cols = np.where(opacity_layer > 0)
        if len(rows) > 0 and len(cols) > 0:
            margin: Tuple[float, float] = (0.1, 0.1)
            width: int = rows.max() - rows.min()
            height: int = cols.max() - cols.min()
            xmin: int = int(max(0, cols.min() - margin[0] * width))
            ymin: int = int(max(0, rows.min() - margin[1] * height))
            xmax: int = int(min(cols.max() + margin[0] * width, alpha.shape[1]))
            ymax: int = int(min(rows.max() + margin[1] * height, alpha.shape[0]))
            if xmin < 0 or ymin < 0 or xmax > alpha.shape[1] or ymax > alpha.shape[0]:
                raise Exception("Error!!! Out of image.")
            return xmin, ymin, xmax, ymax


    def calc_random_move_matrix(self, b_boxes: List[BoundingBox]) -> np.float32:
        """
        Calculate the random move matrix depending on the implant area

        | Output:
        | np.float32 -- move matrix
        """
        inside: bool = True

        if len(b_boxes) != 0:
            while(inside):
                for box in b_boxes:
                    y_padding: int = random.randint(self.Y_MIN_IMPLANT_AREA, self.Y_MAX_IMPLANT_AREA)
                    x_padding: int = random.randint(self.X_MIN_IMPLANT_AREA, self.X_MAX_IMPLANT_AREA)
                    x_pos = self.width/2 + x_padding
                    y_pos = self.height/2 + y_padding
                    if not (box.xmin < x_pos < box.xmax and box.ymin < y_pos < box.ymax):
                        inside = False
        else:
            y_padding: int = random.randint(self.Y_MIN_IMPLANT_AREA, self.Y_MAX_IMPLANT_AREA)
            x_padding: int = random.randint(self.X_MIN_IMPLANT_AREA, self.X_MAX_IMPLANT_AREA)

        return np.float32([[1, 0, x_padding], [0, 1, y_padding]])

    def add_background(self,
                       b_boxes: List[BoundingBox],
                       img_file: Tuple[str, str],
                       bg_img_nat: np.ndarray,
                       bg_img_white: np.ndarray):
        """
        Add background to input image

        | Keyword arguments:
        | img_file -- file name of masks and normal image
        | bg_img_nat -- background image for images
        | bg_img_white -- background image for masks

        | Output:
        | np.ndarray : result images
        | BoundingBox : boudning box with positioning of image
        """
        rgba: np.ndarray = cv2.imread(img_file[0], -1)
        rgba_height: int = rgba.shape[0]
        rgba_width: int = rgba.shape[1]
        scale: float = self.width / self.height
        rgba = self.resize_object(rgba, rgba_width, rgba_height, scale)
        rgba = cv2.resize(rgba, dsize=(self.width, self.height))
        rgb: np.ndarray = rgba[:, :, :3]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        alpha_mask: np.ndarray = rgba[:, :, 3] // 255

        rgba_2: np.ndarray = cv2.imread(img_file[1], -1)
        rgba_height_2: int = rgba_2.shape[0]
        rgba_width_2: int = rgba_2.shape[1]
        scale_2: float = self.width / self.height
        rgba_2 = self.resize_object(rgba_2, rgba_width_2, rgba_height_2, scale_2)
        rgba_2 = cv2.resize(rgba_2, dsize=(self.width, self.height))
        rgb_2: np.ndarray = rgba_2[:, :, :3]
        rgb_2 = cv2.cvtColor(rgb_2, cv2.COLOR_BGR2RGB)
        alpha_mask_2: np.ndarray = rgba_2[:, :, 3] // 255
        tup_1: Tuple = (rgb, rgb_2)
        tup_2: Tuple = (alpha_mask, alpha_mask_2)

        move_matrix: np.ndarray = self.calc_random_move_matrix(b_boxes)
        rgb, alpha_mask, rgb_2, alpha_mask_2 = self.random_scale(tup_1, tup_2, move_matrix)
        tup_1: Tuple = (rgb, rgb_2)
        tup_2: Tuple = (alpha_mask, alpha_mask_2)
        rgb, alpha_mask, rgb_2, alpha_mask_2 = self.random_move(tup_1, tup_2, move_matrix)

        rgb[:, :, 0] = rgb[:, :, 0] * alpha_mask
        rgb[:, :, 1] = rgb[:, :, 1] * alpha_mask
        rgb[:, :, 2] = rgb[:, :, 2] * alpha_mask

        inv_alpha_mask = 1 - alpha_mask
        bg_img_nat_cop = bg_img_nat.copy()
        bg_img_nat_cop[:, :, 0] = bg_img_nat_cop[:, :, 0] * inv_alpha_mask
        bg_img_nat_cop[:, :, 1] = bg_img_nat_cop[:, :, 1] * inv_alpha_mask
        bg_img_nat_cop[:, :, 2] = bg_img_nat_cop[:, :, 2] * inv_alpha_mask
        result_1 = bg_img_nat_cop + rgb
        result_1 = cv2.cvtColor(result_1, cv2.COLOR_BGR2BGRA)


        rgb_2[:, :, 0] = rgb_2[:, :, 0] * alpha_mask_2
        rgb_2[:, :, 1] = rgb_2[:, :, 1] * alpha_mask_2
        rgb_2[:, :, 2] = rgb_2[:, :, 2] * alpha_mask_2

        inv_alpha_mask = 1 - alpha_mask_2
        bg_img_white_cop = bg_img_white.copy()
        bg_img_white_cop[:, :, 0] = bg_img_white_cop[:, :, 0] * inv_alpha_mask
        bg_img_white_cop[:, :, 1] = bg_img_white_cop[:, :, 1] * inv_alpha_mask
        bg_img_white_cop[:, :, 2] = bg_img_white_cop[:, :, 2] * inv_alpha_mask
        result_2 = bg_img_white_cop + rgb_2
        result_2 = cv2.cvtColor(result_2, cv2.COLOR_BGR2BGRA)
        b_box: BoundingBox = BoundingBox(self.get_border(alpha_mask))
        return result_1, result_2, b_box

    def add_multiple_background(self,
                       img_files: List[Tuple[str, str]],
                       bg_img_nat: np.ndarray,
                       bg_img_white: np.ndarray):
        result = []
        b_boxes: List[BoundingBox] = []
        for img_file in img_files:
            res_1, res_2, b_box = self.add_background(b_boxes, img_file, bg_img_nat, bg_img_white)
            result.append((res_1, res_2))
            b_boxes.append(b_box)

        return result
