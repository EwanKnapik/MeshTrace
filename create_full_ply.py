import argparse
import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import trimesh


def save_ply_with_sh(verts, faces, features_dc, features_rest, opacities,
                     sigma_value, active_sh_degree, max_sh_degree, path):
    """
    Save a PLY file with full spherical harmonics data for rendering engines
    (e.g. Unity, Unreal, custom WebGL viewers).

    Vertex properties:
      x, y, z              – position (float32)
      f_dc_0..f_dc_2       – 0th-order (DC) SH coefficients for R, G, B (float32)
      f_rest_0..f_rest_44  – higher-order SH coefficients (float32)
                             Layout: [R_c0..R_c14, G_c0..G_c14, B_c0..B_c14]
                             (follows the 3D Gaussian Splatting convention)
      opacity              – per-vertex opacity in logit space (float32)
                             actual_opacity = sigmoid(opacity)
      sigma                – per-vertex sigma (float32)

    Face properties:
      vertex_indices       – 3 vertex indices per triangle (int32)

    To reconstruct view-dependent color in a shader:
      1. Compute the view direction d = normalize(vertex_pos - camera_pos)
      2. Evaluate SH basis functions Y_l^m(d) up to degree 3
      3. color_c = clamp(sum_i(coeff_c_i * Y_i(d)) + 0.5, 0, 1) for c in {R,G,B}
    """
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    num_dc = features_dc.shape[1]      # 3
    num_rest = features_rest.shape[1]   # 45

    print(f"{num_verts} vertices, {num_faces} faces")
    print(f"SH: {num_dc} DC + {num_rest} rest = {num_dc + num_rest} coefficients "
          f"(degree {max_sh_degree}, active {active_sh_degree})")

    # ── vertex dtype ──
    vert_props = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    for i in range(num_dc):
        vert_props.append((f"f_dc_{i}", "f4"))
    for i in range(num_rest):
        vert_props.append((f"f_rest_{i}", "f4"))
    vert_props.append(("opacity", "f4"))
    vert_props.append(("sigma", "f4"))

    vert_data = np.empty(num_verts, dtype=vert_props)

    # positions
    vert_data["x"] = verts[:, 0]
    vert_data["y"] = verts[:, 1]
    vert_data["z"] = verts[:, 2]

    # DC SH coefficients
    for i in range(num_dc):
        vert_data[f"f_dc_{i}"] = features_dc[:, i]

    # higher-order SH coefficients
    for i in range(num_rest):
        vert_data[f"f_rest_{i}"] = features_rest[:, i]

    # opacity (logit space) and sigma
    vert_data["opacity"] = opacities
    vert_data["sigma"] = np.full(num_verts, sigma_value, dtype=np.float32)

    # ── face dtype ──
    print(f"Faces: {faces}")
    face_dtype = [("vertex_indices", "i4", (3,))]
    face_data = np.empty(num_faces, dtype=face_dtype)
    face_data["vertex_indices"] = faces

    # ── write PLY ──
    vert_el = PlyElement.describe(vert_data, "vertex")
    face_el = PlyElement.describe(face_data, "face")

    ply = PlyData([vert_el, face_el], comments=[
        "triangle_splatting",
        f"active_sh_degree {active_sh_degree}",
        f"max_sh_degree {max_sh_degree}",
        "f_dc: DC SH coefficients (R, G, B)",
        "f_rest: higher-order SH, layout [R0..R14, G0..G14, B0..B14]",
        "opacity: logit space, apply sigmoid for [0,1]",
    ], text=False)
    ply.write(path)

    print(f"Saved {path}")

def create_ply_rgb(path,output_name,instance_trgl=None):
    # ── load checkpoint ──
    sd = torch.load(path, map_location="cpu", weights_only=False)

    # positions & connectivity
    vertices = sd["triangles_points"]          # [V, 3]
    triangle_indices = sd["_triangle_indices"]  # [T, 3]
    if instance_trgl is not None:

        corrected_instance_trgl = [idx for idx in instance_trgl if idx < triangle_indices.shape[0] and idx >= 0]

        # weights is a list of triangle indices belonging to the target object
        #sti = Subset of Triangle Indices
        sti=triangle_indices[corrected_instance_trgl]
        vertices_indices=set(idx.item() for t in sti for idx in t)
        subset_vertices=[]
        Dict_trans={}
        for i in range(vertices.shape[0]):
            if i in vertices_indices:
                subset_vertices.append(i)
                Dict_trans[i]=len(subset_vertices)-1

        orig_idx = torch.tensor(subset_vertices, dtype=torch.long)
        vertices = vertices[orig_idx]

        # Remap triangle indices to the new (compact) vertex indices
        new_triangles = [[Dict_trans[v.item()] for v in T] for T in sti]
        triangle_indices = torch.tensor(new_triangles, dtype=torch.long)
    else:
        orig_idx = None

    # ── spherical harmonics ──
    f_dc   = sd["features_dc"]    # [V, 1, 3]
    if orig_idx is not None:
        f_dc = f_dc[orig_idx]
    
    
    verts_np = vertices.detach().cpu().numpy()
    faces_np = triangle_indices.detach().cpu().numpy()

    # Compute colors (same as original 3D Gaussian Splatting training)
    SH_C0 = 0.28209479177387814
    colors = SH_C0 * f_dc + 0.5
    colors = torch.clamp(colors, 0.0, 1.0)
    colors_u8 = (colors * 255.0).round().to(torch.uint8).cpu().numpy()
    colors_u8 = colors_u8.squeeze()  # Remove the middle dimension [V, 1, 3] -> [V, 3]

    mesh = trimesh.Trimesh(vertices=verts_np.astype(np.float32),
                           faces=faces_np.astype(np.int32),
                           vertex_colors=colors_u8.astype(np.uint8),
                           process=False)
    
    # Export as PLY
    mesh.export(output_name, file_type='ply')


def create_instance_ply_RGB(path,output_name):
    sd = torch.load(path, map_location="cpu", weights_only=False)

    vertices = sd["triangles_points"]
    triangle_indices = sd["_triangle_indices"]
    num_faces = triangle_indices.shape[0]
    num_vertices = vertices.shape[0]

    f_dc = sd["features_dc"]
    SH_C0 = 0.28209479177387814
    colors = SH_C0 * f_dc + 0.5
    colors = torch.clamp(colors, 0.0, 1.0)
    colors_u8 = (colors * 255.0).round().to(torch.uint8).cpu().numpy().squeeze(1)

    # Derive per-face instance ids from checkpoint instance features.
    instance_data = sd.get("instance_feature", None)
    if instance_data is not None:
        instance_tensor = instance_data.detach().cpu()
        faces_ids = torch.argmax(instance_tensor, dim=1).to(torch.int64)
    else:
        faces_ids = torch.zeros(triangle_indices.shape[0], dtype=torch.int64)


    verts_np = vertices.detach().cpu().numpy().astype(np.float32)
    faces_np = triangle_indices.detach().cpu().numpy().astype(np.int32)

    vert_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vert_data = np.empty(verts_np.shape[0], dtype=vert_dtype)
    vert_data["x"] = verts_np[:, 0]
    vert_data["y"] = verts_np[:, 1]
    vert_data["z"] = verts_np[:, 2]
    vert_data["red"] = colors_u8[:, 0]
    vert_data["green"] = colors_u8[:, 1]
    vert_data["blue"] = colors_u8[:, 2]

    # Face schema:
    # element face N
    # property list uchar int vertex_indices
    # property int mesh_id
    # Save one PLY per instance id.
    unique_instance_ids = torch.unique(faces_ids).tolist()
    positive_instance_ids = [int(i) for i in unique_instance_ids if int(i) >= 0]
    if len(positive_instance_ids) > 0:
        instance_ids_to_export = positive_instance_ids
    else:
        instance_ids_to_export = [int(i) for i in unique_instance_ids]

    base, ext = os.path.splitext(output_name)
    if ext == "":
        ext = ".ply"

    scene_dir = path.split('/')
    arg=scene_dir.index("point_cloud")
    scene_dir = "/".join(scene_dir[:arg])
    segment_dir="segmented_instances"
    segment_dir=os.path.join(scene_dir,segment_dir)
    os.makedirs(segment_dir, exist_ok=True)

    files_list=os.listdir(segment_dir)
    idxs=[int(file.split("_")[-1]) for file in files_list]
    if len(idxs)==0:
        new_idx=0
    else:
        new_idx=max(idxs)+1

    export_dir =os.path.join(segment_dir, f"instance_ply_{new_idx}")
    os.makedirs(export_dir, exist_ok=True)

    exported = 0
    for instance_id in instance_ids_to_export:
        mask = (faces_ids == instance_id)
        if mask.sum().item() == 0:
            continue

        selected_faces = triangle_indices[mask]
        vertices_indices = set(idx.item() for t in selected_faces for idx in t)
        subset_vertices = []
        index_map = {}
        for i in range(num_vertices):
            if i in vertices_indices:
                subset_vertices.append(i)
                index_map[i] = len(subset_vertices) - 1

        if len(subset_vertices) == 0:
            continue

        orig_idx = torch.tensor(subset_vertices, dtype=torch.long)
        sub_verts_np = verts_np[orig_idx.numpy()]
        sub_colors_u8 = colors_u8[orig_idx.numpy()]

        remapped_faces = np.array(
            [[index_map[v.item()] for v in tri] for tri in selected_faces],
            dtype=np.int32,
        )

        sub_vert_data = np.empty(sub_verts_np.shape[0], dtype=vert_dtype)
        sub_vert_data["x"] = sub_verts_np[:, 0]
        sub_vert_data["y"] = sub_verts_np[:, 1]
        sub_vert_data["z"] = sub_verts_np[:, 2]
        sub_vert_data["red"] = sub_colors_u8[:, 0]
        sub_vert_data["green"] = sub_colors_u8[:, 1]
        sub_vert_data["blue"] = sub_colors_u8[:, 2]

        face_dtype = [("vertex_indices", "i4", (3,)), ("instance_id", "i4")]
        sub_face_data = np.empty(remapped_faces.shape[0], dtype=face_dtype)
        sub_face_data["vertex_indices"] = remapped_faces
        sub_face_data["instance_id"] = np.full(remapped_faces.shape[0], instance_id, dtype=np.int32)

        instance_output = f"{base}_instance_{instance_id}{ext}"
        ply = PlyData(
            [
                PlyElement.describe(sub_vert_data, "vertex"),
                PlyElement.describe(sub_face_data, "face"),
            ],
            text=False,
        )
        output_name = os.path.join(export_dir, f"{instance_output}")
        ply.write(output_name)
        print(
            f"Saved instance mesh PLY: {instance_output} "
            f"(instance {instance_id}, vertices {sub_verts_np.shape[0]}, faces {remapped_faces.shape[0]})"
        )
        exported += 1

    if exported == 0:
        print("No instance meshes were exported.")
    


def create_ply_sh(path,output_name,instance_trgl=None):
    # ── load checkpoint ──
    sd = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Keys: {list(sd.keys())}")

    # positions & connectivity
    vertices = sd["triangles_points"]          # [V, 3]
    triangle_indices = sd["_triangle_indices"]  # [T, 3]
    if instance_trgl is not None:

        corrected_instance_trgl = [idx for idx in instance_trgl if idx < triangle_indices.shape[0] and idx >= 0]

        print(len(instance_trgl))
        print(len(corrected_instance_trgl))
        print(f"Removed {len(instance_trgl) - len(corrected_instance_trgl)} invalid triangle indices.")
        # weights is a list of triangle indices belonging to the target object
        #sti = Subset of Triangle Indices
        sti=triangle_indices[corrected_instance_trgl]
        vertices_indices=set(idx.item() for t in sti for idx in t)
        subset_vertices=[]
        Dict_trans={}
        for i in range(vertices.shape[0]):
            if i in vertices_indices:
                subset_vertices.append(i)
                Dict_trans[i]=len(subset_vertices)-1

        orig_idx = torch.tensor(subset_vertices, dtype=torch.long)
        vertices = vertices[orig_idx]

        # Remap triangle indices to the new (compact) vertex indices
        new_triangles = [[Dict_trans[v.item()] for v in T] for T in sti]
        triangle_indices = torch.tensor(new_triangles, dtype=torch.long)
    else:
        orig_idx = None
    print(f"Vertices: {vertices.shape}, Triangles: {triangle_indices.shape}")

    # ── spherical harmonics ──
    features_dc   = sd["features_dc"]    # [V, 1, 3]
    features_rest  = sd["features_rest"]  # [V, 15, 3]
    if orig_idx is not None:
        features_dc = features_dc[orig_idx]
        features_rest = features_rest[orig_idx]

    num_sh_coeffs = features_dc.shape[1] + features_rest.shape[1]  # 16
    max_sh_degree = int(np.sqrt(num_sh_coeffs) - 1)                # 3
    active_sh_degree = int(sd.get("active_sh_degree", max_sh_degree))

    print(f"features_dc:   {features_dc.shape}")
    print(f"features_rest: {features_rest.shape}")
    print(f"SH degree: active={active_sh_degree}, max={max_sh_degree}")

    # Flatten SH following the 3DGS convention:
    #   transpose [V, C, 3] -> [V, 3, C]  then flatten -> [V, 3*C]
    #   Gives layout: [R_coeff0..R_coeffN, G_coeff0..G_coeffN, B_coeff0..B_coeffN]
    dc_flat = (features_dc.detach()
               .transpose(1, 2).flatten(start_dim=1)
               .contiguous().cpu().numpy())     # [V, 3]
    rest_flat = (features_rest.detach()
                 .transpose(1, 2).flatten(start_dim=1)
                 .contiguous().cpu().numpy())    # [V, 45]

    # ── opacity (logit space) ──
    vertex_weight = sd["vertex_weight"]  # [V] or [V, 1]
    if vertex_weight.dim() > 1:
        vertex_weight = vertex_weight.squeeze(-1)
    if orig_idx is not None:
        vertex_weight = vertex_weight[orig_idx]
    opacities = vertex_weight.detach().cpu().numpy()

    # ── sigma (global scalar) ──
    sigma_raw = sd["sigma"]
    if isinstance(sigma_raw, torch.Tensor):
        sigma_value = sigma_raw.item()
    else:
        sigma_value = float(sigma_raw)
    print(f"Sigma (raw): {sigma_value}")

    # ── export ──
    save_ply_with_sh(
        verts=vertices.detach().cpu().numpy().astype(np.float32),
        faces=triangle_indices.detach().cpu().numpy().astype(np.int32),
        features_dc=dc_flat.astype(np.float32),
        features_rest=rest_flat.astype(np.float32),
        opacities=opacities.astype(np.float32),
        sigma_value=sigma_value,
        active_sh_degree=active_sh_degree,
        max_sh_degree=max_sh_degree,
        path=output_name,
    )


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
        default="mesh_sh.ply",
        help="Name of the output PLY file (default: mesh_sh.ply)",
    )
    parser.add_argument(
        "--instance",
        type=bool,
        default=False,
        help="WIP",
    )
    args = parser.parse_args()
    if args.instance == True:
        create_instance_ply_RGB(args.path,args.output_name)
    else:
        create_ply_rgb(args.path, args.output_name)


if __name__ == "__main__":
    main()
