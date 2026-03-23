import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement


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
    ply.write("test.ply")

    print(f"Saved {path}")

def create_ply(path,output_name,instance_trgl=None):
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
    args = parser.parse_args()
    create_ply(args.path, args.output_name)


if __name__ == "__main__":
    main()
