# CVProject - Image Inpainting

## Prerequisite

Required environment:
    Anaconda3
    PyTorch
    matplotlib
    torchvision
    piq (use "pip install piq" to install it)

Download dataset and organize the dataset folder as follows:
```
| -- dataset
    | -- train
    | -- val
```

Run `python generate_mask.py` to generate 5000 masks of 256*256 size by default. The corresponding masks are located in ./dataset
You need to choose which masks to use (mask0/mask1/mask2), and copy the corresponding masks into ./dataset/masks
different folders represent different hole-to-image ratios, mask0/mask1/mask2 represent 0-0.1 0.1-0.2 0.2-0.3 Respectively

## Train

Run `python train.py` (with default parameters)

## Test

Run `python test.py` (with default parameters)

## CUDA

In test.py train.py util.py , you can change torch.device ("cpu" or "cuda") to determine the compute device.
