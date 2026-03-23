#!/bin/bash
srun \
  --job-name=triangle_splatting \
  --chdir=/home/KnapikE/mesh-splatting/ \
  --partition=rtx30 \
  --gres=gpu:1 \
  --nodes=1 \
  python extract_objects.py -s ~/input/POLY_01/ -m output/POLY_01/