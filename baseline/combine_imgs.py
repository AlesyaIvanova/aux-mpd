import argparse
from pathlib import Path
import numpy as np

from PIL import Image


def combine_imgs(epoch):
    path_to_folder = f'evaluation/chi_plots/{str(epoch)}'
    path_to_save = f'evaluation/chi_full_plots/{str(epoch)}.png'

    variables = [
        'crossing_angle',
        'dip_angle',
        'drift_length',
        'pad_coord_fraction',
        'time_bin_fraction',
    ]

    stats = [
        'Mean0',
        'Mean1',
        'Sigma0^2',
        'Sigma1^2',
        'Cov01',
        # 'Sum',
    ]

    img_path = Path(path_to_folder)
    images = [[Image.open(img_path / f'{s} vs {v}.png') for v in variables] for s in stats]

    width, height = images[0][0].size

    new_image = Image.new('RGB', (width * len(stats), height * len(variables)))

    x_offset = 0
    for img_line in images:
        y_offset = 0
        for img in img_line:
            new_image.paste(img, (x_offset, y_offset))
            y_offset += img.size[1]
        x_offset += img.size[0]

    new_image.save(path_to_save)
