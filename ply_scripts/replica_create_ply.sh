#!/bin/bash


python create_ply.py output/REPLICA/office_0/Sequence_1/point_cloud/iteration_20000 --out mesh_office_0.ply
python create_ply.py output/REPLICA/office_1/Sequence_1/point_cloud/iteration_30000 --out mesh_office_1.ply
python create_ply.py output/REPLICA/office_2/Sequence_1/point_cloud/iteration_30000 --out mesh_office_2.ply
python create_ply.py output/REPLICA/office_3/Sequence_1/point_cloud/iteration_30000 --out mesh_office_3.ply
python create_ply.py output/REPLICA/office_4/Sequence_1/point_cloud/iteration_30000 --out mesh_office_4.ply
python create_ply.py output/REPLICA/room_0/Sequence_1/point_cloud/iteration_30000 --out mesh_room_0.ply
python create_ply.py output/REPLICA/room_1/Sequence_1/point_cloud/iteration_30000 --out mesh_room_1.ply
python create_ply.py output/REPLICA/room_2/Sequence_1/point_cloud/iteration_30000 --out mesh_room_2.ply