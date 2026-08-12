#!/bin/bash


iter=chkpt_1000


python create_ply.py output/custom/Detailed\ Drum\ Set/point_cloud/iteration_$iter/ --out drums.ply
python create_ply.py output/custom/indoor\ plant\ ficus/point_cloud/iteration_$iter/ --out ficus.ply
python create_ply.py output/custom/Microphone/point_cloud/iteration_$iter/ --out Microphone.ply
python create_ply.py output/custom/Hotdog/point_cloud/iteration_$iter/ --out hotdog.ply


mv *ply plys/
