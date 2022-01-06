from typing import Tuple


class BoundingBox:
    def __init__(self, box: Tuple[int, int, int, int]):
        """
        | Keyword arguments:
        | Tuple
        |    int : xmin
        |    int : ymin
        |    int : xmax
        |    int : ymax
        """
        self.xmin: int = box[0]
        self.ymin: int = box[1]
        self.xmax: int = box[2]
        self.ymax: int = box[3]
