import argparse
import numpy as np
import random
from PIL import Image

walk_dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def write_masks(mask,mask_type):
    if(mask_type == "-0.1"):
        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format('dataset/mask0', i))

    elif(mask_type == "0.1-0.2"):
        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format('dataset/mask1', i))

    elif(mask_type == "0.2-0.3"):
        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format('dataset/mask2', i))

    else:
        print("invalid canvas!")
    
def write_masks_train(mask):
    img = Image.fromarray(mask * 255).convert('1')
    img.save('{:s}/{:06d}.jpg'.format('dataset/masks', i))


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
    zeros = 0
    canvas[np.array(x_list), np.array(y_list)] = 0
    for i in canvas:
        zeros += np.bincount(i)[0]
    value = zeros/length
    
    if(value <0.1):
        mask_type = "-0.1"
    elif(value < 0.2):
        mask_type = "0.1-0.2"
    elif(value < 0.3): 
        mask_type = "0.2-0.3"
    else:
        mask_type = "invalid"

    return canvas,mask_type
    


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--out_dir', type=str, default='dataset/masks')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for i in range(0, 3):
        if not os.path.exists('dataset/mask{}'.format(i)):
            os.makedirs('dataset/mask{}'.format(i))

    for i in range(args.N):
        canvas = np.ones((args.size, args.size)).astype(np.int8)
        mask,mask_type = random_walk(canvas)
        write_masks_train(mask)
        write_masks(mask,mask_type)
        print("Process: ", i)

