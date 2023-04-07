from PIL import Image
from torchvision.transforms import ToPILImage


def get_img_grid(imgs, rows, cols):
    """Generates and saves a grid of images given an array of PIL Images.

    Args:
        imgs ([PIL.Image])
        rows (int): Number of rows in grid
        cols (int): Number of columns in grid
    """
    # print(imgs.size())
    _, w, h = imgs[0].size()
    grid = Image.new('L', size=(cols*w, rows*h))
    # grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(ToPILImage()(img), box=(i % cols*w, i//cols*h))

    return grid
