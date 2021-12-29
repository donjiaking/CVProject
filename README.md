# CVProject - Image Inpainting Using Partial Convolution

## Prerequisite

Required environment:
    Numpy
    PyTorch
    matplotlib
    cv2
    PIL
    piq (use "pip install piq" to install it)

First download dataset and organize the dataset folder as follows:
```
| -- dataset
    | -- train
    | -- test
    | -- val
```

Then run `python generate_mask.py` to generate 10000 masks of 256*256 size by default. It will create a directory `./dataset/masks`, which is used for training, and `./dataset/mask0`, `./dataset/mask1`, `./dataset/mask2`, which are used for testing under different kind of masks.

Note: Different folders represent different hole-to-image ratios, mask0 mask1 mask2 represent 0-0.1 0.1-0.2 0.2-0.3 Respectively.

## Train

Run `python train.py` (with default parameters)

You can set customized parameters: `python train.py --init_lr 2e-3 --batch_size 4 --epochs 1 --img_dir ./dataset/train --mask_dir ./dataset/masks --model_dir ./model --out_dir ./result`

## Test

Run `python test.py` (with default parameters)

You can set customized parameters: `python test.py --batch_size 16 --img_dir ./dataset/test --mask_dir ./dataset/masks --model_dir ./model --out_dir ./result`

## CUDA

In test.py train.py util.py , you can change `torch.device ("cpu" or "cuda")` to determine the compute device.
