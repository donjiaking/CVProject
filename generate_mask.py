import argparse
import numpy as np
import random
from PIL import Image

walk_dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

"""
use random walk to generate 0-1 mask
"""
def random_walk(canvas):
    img_size = canvas.shape[0]
    length = img_size ** 2
    x = random.randint(0, img_size - 1)
    y = random.randint(0, img_size - 1)
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(walk_dir) - 1)
        x = np.clip(x + walk_dir[r][0], a_min=0, a_max=img_size - 1)
        y = np.clip(y + walk_dir[r][1], a_min=0, a_max=img_size - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--N', type=int, default=5000)
    parser.add_argument('--out_dir', type=str, default='dataset/masks')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(args.N):
        canvas = np.ones((args.size, args.size)).astype(np.int8)
        mask = random_walk(canvas)
        print("Process: ", i)

        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format(args.out_dir, i))