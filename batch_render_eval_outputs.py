import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


BLENDER_RENDER_ONE_SCRIPT = r'''
import argparse
import json
import math
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
    import sys

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--env-names", default="map1,map2,map3,map4,map5")
    parser.add_argument("--env-extension", default="exr")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
    parser.add_argument("--cycles-device", default="CPU", choices=["GPU", "CPU"])
    parser.add_argument("--env-strength", type=float, default=1.0)
    parser.add_argument("--background-color", default="127,127,127")
    parser.add_argument("--lens", type=float, default=70.0)
    parser.add_argument("--camera-distance", type=float, default=2.3)
    parser.add_argument("--ortho", action="store_true")
    parser.add_argument("--ortho-scale", type=float, default=1.35)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_rgb(text):
    values = [int(float(item.strip())) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--background-color must have exactly 3 values.")
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
        print(f"Warning: {device} is not valid for Cycles here; falling back to CPU.")
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
    return envs


def main():
    args = parse_args()
    bg_rgb = parse_rgb(args.background_color)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = resolve_envs(args.env_dir, args.env_names, args.env_extension)

    clear_scene()
    mesh_objects = import_mesh(args.mesh)
    meta = normalize_objects(mesh_objects)
    cam = setup_scene(args)

    frames = []
    for env in envs:
        set_world_env(env["path"], args.env_strength, bg_rgb, args.transparent)
        for view_name, azim, elev in VIEW_SPECS:
            matrix = set_camera(cam, azim, elev, args.camera_distance)
            flat_name = f"{view_name}_{args.resolution}__{env['name']}.png"
            out_path = out_dir / flat_name
            if out_path.exists() and not args.overwrite:
                print(f"Skip existing: {out_path}")
            else:
                bpy.context.scene.render.filepath = str(out_path)
                bpy.ops.render.render(write_still=True)
            frames.append(
                {
                    "file_path": flat_name,
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

    manifest = {
        "mesh": str(Path(args.mesh).resolve()),
        "out_dir": str(out_dir.resolve()),
        "resolution": args.resolution,
        "background_color": bg_rgb,
        "transparent": args.transparent,
        "normalization": meta,
        "frames": frames,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rendered {len(frames)} images to {out_dir}")


if __name__ == "__main__":
    main()
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-render generated MVP evaluation meshes with the same fixed 30-view protocol as GT."
    )
    parser.add_argument("--eval-cases", required=True, help="Benchmark eval_cases.json.")
    parser.add_argument("--generated-root", required=True, help="Root produced by batch_generate_eval_meshes.py.")
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to render, e.g. base ft rl. Default: detect.")
    parser.add_argument("--mesh-name", default="textured_mesh.glb")
    parser.add_argument("--out-subdir", default="renders")
    parser.add_argument("--env-names", default="map1,map2,map3,map4,map5")
    parser.add_argument("--env-extension", default="exr")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
    parser.add_argument("--cycles-device", default="CPU", choices=["GPU", "CPU"])
    parser.add_argument("--env-strength", type=float, default=1.0)
    parser.add_argument("--background-color", default="127,127,127")
    parser.add_argument("--lens", type=float, default=70.0)
    parser.add_argument("--camera-distance", type=float, default=2.3)
    parser.add_argument("--ortho", action="store_true")
    parser.add_argument("--ortho-scale", type=float, default=1.35)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=0, help="Per Blender job timeout in seconds. 0 means no timeout.")
    parser.add_argument("--script-path", default=None, help="Where to write the temporary Blender render script.")
    return parser.parse_args()


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("--eval-cases must be a JSON list.")
    return cases


def expected_png_count(out_dir, env_names):
    total = 0
    for env_name in env_names:
        total += len(list(out_dir.glob(f"*__{env_name}.png")))
    return total


def detect_methods(generated_root, cases):
    if not cases:
        return []
    first_case_dir = Path(generated_root) / cases[0]["name"]
    if not first_case_dir.exists():
        raise FileNotFoundError(first_case_dir)
    methods = []
    for path in sorted(first_case_dir.iterdir()):
        if path.is_dir() and (path / "textured_mesh.glb").exists():
            methods.append(path.name)
    if not methods:
        raise RuntimeError(f"No method folders with textured_mesh.glb found under {first_case_dir}")
    return methods


def write_blender_script(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BLENDER_RENDER_ONE_SCRIPT, encoding="utf-8")
    return path


def build_jobs(args, cases, methods):
    generated_root = Path(args.generated_root)
    env_names = [item.strip() for item in args.env_names.split(",") if item.strip()]
    jobs = []
    missing = []
    for case in cases:
        case_name = case["name"]
        for method in methods:
            method_dir = generated_root / case_name / method
            mesh_path = method_dir / args.mesh_name
            out_dir = method_dir / args.out_subdir
            manifest_path = out_dir / "render_manifest.json"
            expected_count = 6 * len(env_names)
            if not mesh_path.exists():
                missing.append(str(mesh_path))
                continue
            if (
                manifest_path.exists()
                and not args.overwrite
                and expected_png_count(out_dir, env_names) >= expected_count
            ):
                print(f"Skip existing: {out_dir}")
                continue
            jobs.append(
                {
                    "case": case_name,
                    "method": method,
                    "mesh": str(mesh_path),
                    "out_dir": str(out_dir),
                }
            )
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing generated meshes, first entries:\n{preview}")
    return jobs


def run_job(args, blender_script, job):
    cmd = [
        args.blender_bin,
        "--factory-startup",
        "-b",
        "-P",
        str(blender_script),
        "--",
        "--mesh",
        job["mesh"],
        "--out-dir",
        job["out_dir"],
        "--env-dir",
        args.env_dir,
        "--env-names",
        args.env_names,
        "--env-extension",
        args.env_extension,
        "--resolution",
        str(args.resolution),
        "--samples",
        str(args.samples),
        "--engine",
        args.engine,
        "--cycles-device",
        args.cycles_device,
        "--env-strength",
        str(args.env_strength),
        "--background-color",
        args.background_color,
        "--lens",
        str(args.lens),
        "--camera-distance",
        str(args.camera_distance),
        "--ortho-scale",
        str(args.ortho_scale),
    ]
    if args.ortho:
        cmd.append("--ortho")
    if args.transparent:
        cmd.append("--transparent")
    if args.overwrite:
        cmd.append("--overwrite")

    timeout = None if args.timeout <= 0 else args.timeout
    print(f"[render] {job['case']} / {job['method']}")
    subprocess.run(cmd, check=True, timeout=timeout)
    return job


def main():
    args = parse_args()
    cases = load_cases(args.eval_cases)
    methods = args.methods or detect_methods(args.generated_root, cases)
    script_path = args.script_path or str(Path(args.generated_root) / "_render_eval_output_one.py")
    blender_script = write_blender_script(script_path)
    jobs = build_jobs(args, cases, methods)
    print(f"Methods: {methods}")
    print(f"Jobs to render: {len(jobs)}")
    if not jobs:
        print("Nothing to render.")
        return

    if args.max_workers <= 1:
        done = [run_job(args, blender_script, job) for job in jobs]
    else:
        done = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_job = {executor.submit(run_job, args, blender_script, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    done.append(future.result())
                except Exception as exc:
                    print(f"[failed] {job['case']} / {job['method']}: {exc}", file=sys.stderr)
                    raise

    summary = {
        "eval_cases": str(Path(args.eval_cases).resolve()),
        "generated_root": str(Path(args.generated_root).resolve()),
        "methods": methods,
        "env_dir": str(Path(args.env_dir).resolve()),
        "env_names": args.env_names,
        "resolution": args.resolution,
        "rendered_jobs": done,
    }
    summary_path = Path(args.generated_root) / "render_outputs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
