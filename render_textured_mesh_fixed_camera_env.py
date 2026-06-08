import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


VIEW_ROTATIONS_DEGREES_XYZ = {
    "front": (0.0, 0.0, 0.0),
    "back": (0.0, 0.0, 180.0),
    "left": (0.0, 0.0, 90.0),
    "right": (0.0, 0.0, -90.0),
    "top": (90.0, 0.0, 0.0),
    "bottom": (-90.0, 0.0, 0.0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render a PBR mesh under a fixed world-space environment.")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--lens", type=float, default=45.0)
    parser.add_argument("--camera-location", type=float, nargs=3, default=(0.0, -3.5, 0.35))
    parser.add_argument("--background-color", type=float, nargs=3, default=(127.0, 127.0, 127.0))
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    return parser.parse_args(script_args)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(mesh_path):
    suffix = Path(mesh_path).suffix.lower()
    if suffix in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(mesh_path))
    else:
        raise ValueError(f"Unsupported mesh format: {suffix}")

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh objects were imported.")
    return mesh_objects


def normalize_scene(mesh_objects):
    bpy.context.view_layer.update()
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)

    center = (min_corner + max_corner) * 0.5
    size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)
    scale = 2.0 / size if size > 0 else 1.0

    root = bpy.data.objects.new("normalized_mesh_root", None)
    bpy.context.collection.objects.link(root)
    for obj in mesh_objects:
        obj.parent = root

    root.location = -center
    bpy.context.view_layer.update()
    root.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return root


def setup_render(resolution, samples):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"


def setup_world_environment(env_path, strength, background_color):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputWorld")
    light_path = nodes.new(type="ShaderNodeLightPath")
    mix = nodes.new(type="ShaderNodeMixShader")
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(str(env_path))
    try:
        env_tex.image.colorspace_settings.name = "Non-Color"
    except TypeError:
        pass

    env_background = nodes.new(type="ShaderNodeBackground")
    env_background.inputs["Strength"].default_value = strength
    camera_background = nodes.new(type="ShaderNodeBackground")
    camera_background.inputs["Color"].default_value = (
        background_color[0] / 255.0,
        background_color[1] / 255.0,
        background_color[2] / 255.0,
        1.0,
    )
    camera_background.inputs["Strength"].default_value = 1.0

    links.new(env_tex.outputs["Color"], env_background.inputs["Color"])
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(env_background.outputs["Background"], mix.inputs[1])
    links.new(camera_background.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_camera(location, lens):
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.location = location
    camera.data.lens = lens
    camera.data.sensor_width = 32.0
    look_at(camera, (0.0, 0.0, 0.0))
    return camera


def render_views(root, camera, out_dir, mesh_path, env_path, strength):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}

    for name, rotation_degrees in VIEW_ROTATIONS_DEGREES_XYZ.items():
        root.rotation_euler = tuple(math.radians(v) for v in rotation_degrees)
        bpy.context.view_layer.update()

        output_path = out_dir / f"{name}.png"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

        metadata[name] = {
            "image": str(output_path),
            "mesh": str(mesh_path),
            "env": str(env_path),
            "env_strength": strength,
            "camera_location": list(camera.location),
            "camera_rotation_euler": list(camera.rotation_euler),
            "lens": camera.data.lens,
            "sensor_width": camera.data.sensor_width,
            "mesh_root_rotation_degrees_xyz": list(rotation_degrees),
            "mesh_root_rotation_euler": list(root.rotation_euler),
        }

    (out_dir / "transforms.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    clear_scene()
    mesh_path = Path(args.mesh).resolve()
    env_path = Path(args.env).resolve()
    out_dir = Path(args.out_dir).resolve()

    mesh_objects = import_mesh(mesh_path)
    root = normalize_scene(mesh_objects)
    setup_render(args.resolution, args.samples)
    setup_world_environment(env_path, args.strength, args.background_color)
    camera = create_camera(tuple(args.camera_location), args.lens)
    render_views(root, camera, out_dir, mesh_path, env_path, args.strength)


if __name__ == "__main__":
    main()
