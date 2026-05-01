import argparse
import os
import numpy as np
from create_full_ply import create_instance_ply_RGB
from utils.system_utils import mkdir_p






def process_bulk_ply(path: str, output_name: str):
    files_list=os.listdir(path)
    idxs=[]
    for file in files_list:
        idxs.append(int(file.split("_")[-1].split(".")[0]) if file.split("_")[-1].split(".")[0].isnumeric() else 0)

    for file in files_list:
        print(file)
        create_instance_ply_RGB(os.path.join(path,file), output_name)















def main():
    parser = argparse.ArgumentParser(
        description="Convert a triangle-splatting checkpoint to a PLY file with "
                    "full spherical harmonics data for use in rendering engines."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the input checkpoint file (e.g., point_cloud_state_dict.pt)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="mesh_sh",
        help="Name of the output PLY file (default: mesh_sh.ply)",
    )
    args = parser.parse_args()
    process_bulk_ply(args.path, args.output_name)


if __name__ == "__main__":
    main()