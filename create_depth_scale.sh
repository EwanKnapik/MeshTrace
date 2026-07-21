#!/bin/bash

python utils/make_depth_scale.py --base_dir ~/input/REPLICA/office_0/Sequence_1/ --depths_dir ~/input/REPLICA/office_0/Sequence_1/depth
python utils/make_depth_scale.py --base_dir ~/input/REPLICA/office_1/Sequence_1/ --depths_dir ~/input/REPLICA/office_1/Sequence_1/depth
python utils/make_depth_scale.py --base_dir ~/input/REPLICA/office_2/Sequence_1/ --depths_dir ~/input/REPLICA/office_2/Sequence_1/depth
python utils/make_depth_scale.py --base_dir ~/input/REPLICA/office_3/Sequence_1/ --depths_dir ~/input/REPLICA/office_3/Sequence_1/depth
python utils/make_depth_scale.py --base_dir ~/input/REPLICA/office_4/Sequence_1/ --depths_dir ~/input/REPLICA/office_4/Sequence_1/depth
python utils/make_depth_scale.py --base_dir ~/input/REPLICA/room_0/Sequence_1/ --depths_dir ~/input/REPLICA/room_0/Sequence_1/depth
python utils/make_depth_scale.py --base_dir ~/input/REPLICA/room_1/Sequence_1/ --depths_dir ~/input/REPLICA/room_1/Sequence_1/depth
python utils/make_depth_scale.py --base_dir ~/input/REPLICA/room_2/Sequence_1/ --depths_dir ~/input/REPLICA/room_2/Sequence_1/depth