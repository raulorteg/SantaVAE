import numpy as np
from PIL import Image


def image_grid(imgs, rows, cols, height=50, width=50):
    """Creates a grid of images in from a list of PIL.Image objects and returns the new grid Image
    :param imgs: list of PIL.Image objects (the Images to be part of the grid)
    :type imgs: list of PIL.Image objects
    :param rows: number of rows to use for the grid
    :type rows: int
    :param cols: number of columns to use for the grid
    :type cols: int
    :param height: height in pixels of each individual image part of the grid
    :type height: int
    :param width: width in pixels of each individual image part of the grid
    :type width: int
    """
    assert len(imgs) == rows * cols

    w, h = height, width
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w + 1, i // cols * h + 1))
    return grid
