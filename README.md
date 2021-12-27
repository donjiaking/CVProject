# CVProject - Image Inpainting

## Prerequisite

Download dataset and organize the dataset folder as follows:
```
| -- dataset
    | -- train
    | -- val
```

Run `python generate_mask.py` to generate 5000 masks of 256*256 size by default.

## Train

Run `python train.py` (with default parameters)

## Test

Run `python test.py` (with default parameters)

## CUDA

In test.py train.py util.py , you can change torch.device ("cpu" or "cuda") to determine the compute device.
