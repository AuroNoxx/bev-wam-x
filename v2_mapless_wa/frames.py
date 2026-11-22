"""Load only the allowed actor observations from one synchronized frame."""
import cv2
import numpy as np
from .geometry import NAMES


def load_observation(folder):
    views = []
    for name in NAMES:
        image = cv2.imread(str(folder / (name + '.png')), cv2.IMREAD_COLOR)
        if image is None or image.shape not in ((224,384,3),(448,768,3)):
            raise ValueError(f'Missing or invalid camera {name} at {folder}')
        views.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with np.load(folder / 'input.npz') as navigation, np.load(folder / 'lidar.npz') as lidar:
        return dict(rgb=(np.stack(views) if len({v.shape for v in views})==1 else tuple(views)), lidar=lidar['bev'], sd_nav=navigation['sd_nav'],
                    cmd_one_hot=navigation['cmd_one_hot'])
