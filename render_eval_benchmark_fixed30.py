import argparse
import json
import math
import shutil
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


VIEW_SPECS = [
    ("front", 0.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", -90.0, 0.0),
    ("right", 90.0, 0.0),
    ("top", 0.0, 90.0),
    ("bottom", 0.0, -90.0),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a fixed 30-image benchmark set for MaterialMVP evaluation. "
            "Run with Blender: blender -b --python render_eval_benchmark_fixed30.py -- ..."
        )
    )
    parser.add_argument("--mesh-dir", required=True, help="Directory containing held-out .glb/.gltf/.obj meshes.")
    parser.add_argument("--out-dir", required=True, help="Output benchmark directory.")
    parser.add_argument("--env-dir", required=True, help="Directory containing map1.exr ... map5.exr or HDR/EXR files.")
    parser.add_argument("--env-names", default="map1,map2,map3,map4,map5")
    parser.add_argument("--env-extension", default="exr")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
    parser.add_argument("--cycles-device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--env-strength", type=float, default=1.0)
    parser.add_argument("--background-color", default="127,127,127", help="Camera-visible RGB background.")
    parser.add_argument("--lens", type=float, default=70.0)
    parser.add_argument("--camera-distance", type=float, default=2.3)
    parser.add_argument("--ortho", action="store_true")
    parser.add_argument("--ortho-scale", type=float, default=1.35)
    parser.add_argument("--transparent", action="store_true", help="Save RGBA with transparent background.")
    parser.add_argument("--case-prefix", default="case")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all meshes.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mesh", action="store_true", help="Copy source mesh into each case directory.")
    parser.add_argument("--reference-key", default="front_1024__map1.png")
    parser.add_argument("--eval-cases-name", default="eval_cases.json")

    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def parse_rgb(text):
    values = [int(float(item.strip())) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--background-color must have exactly 3 comma-separated values.")
    return tuple(max(0, min(255, value)) for value in values)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures]:
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def import_mesh(mesh_path):
    suffix = Path(mesh_path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(mesh_path))
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")
    return mesh_objects


def scene_bounds(mesh_objects):
    points = []
    for obj in mesh_objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    mins = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    maxs = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return mins, maxs


def normalize_objects(mesh_objects):
    mins, maxs = scene_bounds(mesh_objects)
    center = (mins + maxs) * 0.5
    extent = maxs - mins
    max_dim = max(extent.x, extent.y, extent.z)
    if max_dim <= 0:
        raise RuntimeError("Invalid mesh bounds.")
    scale = 1.0 / max_dim
    transform = Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-center)
    for obj in mesh_objects:
        obj.matrix_world = transform @ obj.matrix_world
    new_mins, new_maxs = scene_bounds(mesh_objects)
    return {
        "aabb": [[new_mins.x, new_mins.y, new_mins.z], [new_maxs.x, new_maxs.y, new_maxs.z]],
        "scale": scale,
        "offset": [-center.x, -center.y, -center.z],
        "source_aabb": [[mins.x, mins.y, mins.z], [maxs.x, maxs.y, maxs.z]],
    }


def setup_cycles_device(device):
    scene = bpy.context.scene
    try:
        scene.cycles.device = device
    except TypeError:
        print(f"Warning: {device} is not a valid Cycles device here; falling back to CPU.")
        scene.cycles.device = "CPU"
        return
    if device != "GPU":
        return
    prefs = bpy.context.preferences.addons.get("cycles")
    if not prefs:
        return
    cprefs = prefs.preferences
    for compute_type in ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "NONE"]:
        try:
            cprefs.compute_device_type = compute_type
            cprefs.get_devices()
            enabled = 0
            for dev in cprefs.devices:
                dev.use = dev.type != "CPU"
                enabled += int(dev.use)
            if enabled:
                print(f"Cycles GPU compute device: {compute_type}")
                return
        except Exception:
            continue


def setup_scene(args):
    scene = bpy.context.scene
    scene.render.engine = args.engine
    if args.engine == "CYCLES":
        setup_cycles_device(args.cycles_device)
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(16, args.samples)

    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = bool(args.transparent)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.color_mode = "RGBA" if args.transparent else "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens = args.lens
    cam.data.type = "ORTHO" if args.ortho else "PERSP"
    cam.data.ortho_scale = args.ortho_scale
    return cam


def direction_from_angles(azim_deg, elev_deg):
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    return Vector((math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)))


def set_camera(cam, azim_deg, elev_deg, distance):
    direction = direction_from_angles(azim_deg, elev_deg)
    cam.location = direction * distance
    target = Vector((0.0, 0.0, 0.0))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    return cam.matrix_world.copy()


def set_world_env(env_path, strength, bg_rgb, transparent):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputWorld")
    env_bg = nodes.new("ShaderNodeBackground")
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(str(env_path), check_existing=True)
    env_bg.inputs["Strength"].default_value = strength
    links.new(env_tex.outputs["Color"], env_bg.inputs["Color"])

    if transparent:
        links.new(env_bg.outputs["Background"], out.inputs["Surface"])
        return

    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    cam_bg = nodes.new("ShaderNodeBackground")
    cam_bg.inputs["Color"].default_value = (bg_rgb[0] / 255.0, bg_rgb[1] / 255.0, bg_rgb[2] / 255.0, 1.0)
    cam_bg.inputs["Strength"].default_value = 1.0
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(env_bg.outputs["Background"], mix.inputs[1])
    links.new(cam_bg.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])


def collect_meshes(mesh_dir):
    mesh_dir = Path(mesh_dir)
    meshes = []
    for suffix in ("*.glb", "*.gltf", "*.obj"):
        meshes.extend(mesh_dir.glob(suffix))
    return sorted(meshes)


def resolve_envs(env_dir, env_names, extension):
    env_dir = Path(env_dir)
    envs = []
    for env_name in [item.strip() for item in env_names.split(",") if item.strip()]:
        candidates = [
            env_dir / f"{env_name}.{extension}",
            env_dir / f"{env_name}.exr",
            env_dir / f"{env_name}.hdr",
            env_dir / env_name / f"{env_name}.{extension}",
            env_dir / env_name / f"{env_name}.exr",
            env_dir / env_name / f"{env_name}.hdr",
        ]
        for candidate in candidates:
            if candidate.exists():
                envs.append({"name": env_name, "path": candidate})
                break
        else:
            raise FileNotFoundError(f"Could not find envmap for {env_name} under {env_dir}")
    if not envs:
        raise ValueError("No env maps selected.")
    return envs


def render_case(mesh_path, case_name, case_dir, args, envs, bg_rgb):
    case_dir.mkdir(parents=True, exist_ok=True)
    gt_renders = case_dir / "gt_renders"
    gt_renders.mkdir(parents=True, exist_ok=True)

    clear_scene()
    mesh_objects = import_mesh(mesh_path)
    meta = normalize_objects(mesh_objects)
    cam = setup_scene(args)

    frames = []
    for env in envs:
        set_world_env(env["path"], args.env_strength, bg_rgb, args.transparent)
        for view_name, azim, elev in VIEW_SPECS:
            matrix = set_camera(cam, azim, elev, args.camera_distance)
            flat_name = f"{view_name}_{args.resolution}__{env['name']}.png"
            out_path = gt_renders / flat_name
            if out_path.exists() and not args.overwrite:
                print(f"Skip existing: {out_path}")
            else:
                bpy.context.scene.render.filepath = str(out_path)
                bpy.ops.render.render(write_still=True)
            frames.append(
                {
                    "file_path": f"gt_renders/{flat_name}",
                    "view": view_name,
                    "azimuth": math.radians(azim),
                    "elevation": math.radians(elev),
                    "env_name": env["name"],
                    "env_path": str(env["path"]),
                    "camera_distance": args.camera_distance,
                    "camera_angle_x": cam.data.angle,
                    "proj_type": 1 if args.ortho else 0,
                    "transform_matrix": [list(row) for row in matrix],
                }
            )

    reference_src = gt_renders / args.reference_key
    if not reference_src.exists():
        fallback = gt_renders / f"front_{args.resolution}__{envs[0]['name']}.png"
        reference_src = fallback
    reference_path = case_dir / "reference.png"
    shutil.copy2(reference_src, reference_path)

    if args.copy_mesh:
        copied_mesh = case_dir / f"gt_mesh{Path(mesh_path).suffix.lower()}"
        shutil.copy2(mesh_path, copied_mesh)
        mesh_for_manifest = copied_mesh
    else:
        mesh_for_manifest = Path(mesh_path).resolve()

    manifest = {
        "name": case_name,
        "source_mesh": str(Path(mesh_path).resolve()),
        "mesh": str(mesh_for_manifest),
        "reference": str(reference_path),
        "gt_renders": str(gt_renders),
        "resolution": args.resolution,
        "background_color": bg_rgb,
        "transparent": args.transparent,
        "normalization": meta,
        "frames": frames,
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    args = parse_args()
    bg_rgb = parse_rgb(args.background_color)
    mesh_paths = collect_meshes(args.mesh_dir)
    if args.max_cases > 0:
        mesh_paths = mesh_paths[: args.max_cases]
    if not mesh_paths:
        raise RuntimeError(f"No meshes found in {args.mesh_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = resolve_envs(args.env_dir, args.env_names, args.env_extension)
    eval_cases = []
    benchmark = {
        "mesh_dir": str(Path(args.mesh_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "envs": [{"name": env["name"], "path": str(env["path"])} for env in envs],
        "cases": [],
    }

    for idx, mesh_path in enumerate(mesh_paths, start=args.start_index):
        case_name = f"{args.case_prefix}{idx:04d}"
        print(f"\nRendering {case_name}: {mesh_path}")
        case_dir = out_dir / case_name
        manifest = render_case(mesh_path, case_name, case_dir, args, envs, bg_rgb)
        benchmark["cases"].append(manifest)
        eval_cases.append(
            {
                "name": case_name,
                "mesh": manifest["mesh"],
                "reference": manifest["reference"],
                "gt_renders": manifest["gt_renders"],
            }
        )
        (out_dir / args.eval_cases_name).write_text(json.dumps(eval_cases, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "benchmark_manifest.json").write_text(
            json.dumps(benchmark, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nDone. Eval cases: {out_dir / args.eval_cases_name}")
    print(f"Benchmark manifest: {out_dir / 'benchmark_manifest.json'}")


if __name__ == "__main__":
    main()
