import argparse
import copy
import glob
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from einops import rearrange
from PIL import Image

try:
    from utils.torchvision_fix import apply_fix

    apply_fix()
except Exception as exc:
    print(f"Warning: torchvision compatibility fix was not applied: {exc}")

from materialmvp.lora_utils import inject_lora_into_attention, save_lora_checkpoint
from materialmvp.pipeline import to_rgb_image
from materialmvp.rl.ddim_with_logprob import ddim_step_with_logprob


BLENDER_RENDER_SCRIPT = r'''
import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv if "sys" in globals() else __import__("sys").argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--camera-distance-scale", type=float, default=2.4)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(mesh_path):
    suffix = Path(mesh_path).suffix.lower()
    if suffix == ".glb" or suffix == ".gltf":
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=mesh_path)
        else:
            bpy.ops.import_scene.obj(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")
    return mesh_objects


def scene_bounds(objects):
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    min_v = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (min_v + max_v) * 0.5
    radius = max((p - center).length for p in points)
    return center, max(radius, 1e-4)


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def set_world_hdri(hdri_path, strength, bg_color):
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputWorld")
    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    env_bg = nodes.new("ShaderNodeBackground")
    cam_bg = nodes.new("ShaderNodeBackground")
    env_bg.inputs["Strength"].default_value = strength
    cam_bg.inputs["Color"].default_value = bg_color
    cam_bg.inputs["Strength"].default_value = 1.0
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(env_bg.outputs["Background"], mix.inputs[1])
    links.new(cam_bg.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])

    if hdri_path:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path)
        links.new(env.outputs["Color"], env_bg.inputs["Color"])
    else:
        env_bg.inputs["Color"].default_value = bg_color


def add_fallback_lights(center, radius):
    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y - radius * 2.0, center.z + radius * 2.5))
    key = bpy.context.object
    key.name = "Reward_Key_Area"
    key.data.energy = 450.0
    key.data.size = radius * 2.0
    bpy.ops.object.light_add(type="POINT", location=(center.x + radius * 1.5, center.y + radius * 1.2, center.z + radius))
    fill = bpy.context.object
    fill.name = "Reward_Fill_Point"
    fill.data.energy = 70.0


def configure_renderer(args):
    scene = bpy.context.scene
    scene.render.engine = args.engine
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        try:
            scene.cycles.device = "GPU"
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            for device in bpy.context.preferences.addons["cycles"].preferences.devices:
                device.use = True
        except Exception:
            pass


def fit_distance_for_mesh(radius, camera, margin, min_distance_scale):
    fov = min(float(camera.data.angle_x), float(camera.data.angle_y))
    fov = max(fov, math.radians(5.0))
    fit_distance = radius * margin / max(math.sin(fov * 0.5), 1e-4)
    min_distance = radius * min_distance_scale
    return max(fit_distance, min_distance)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    clear_scene()
    mesh_objects = import_mesh(args.mesh)
    center, radius = scene_bounds(mesh_objects)
    configure_renderer(args)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.data.lens = float(plan.get("lens", 50.0))
    bg_rgb = plan.get("background_color", [127, 127, 127])
    bg_color = (
        float(bg_rgb[0]) / 255.0,
        float(bg_rgb[1]) / 255.0,
        float(bg_rgb[2]) / 255.0,
        1.0,
    )
    framing_margin = float(plan.get("framing_margin", 1.25))

    hdri_items = plan.get("hdris", [])
    if not hdri_items:
        hdri_items = [{"name": "default", "path": None, "strength": plan.get("world_strength", 1.0)}]

    render_records = []
    views = plan["views"]
    for hdri_idx, hdri in enumerate(hdri_items):
        hdri_path = hdri.get("path")
        set_world_hdri(hdri_path, float(hdri.get("strength", plan.get("world_strength", 1.0))), bg_color)
        if not hdri_path:
            add_fallback_lights(center, radius)
        for view_idx, view in enumerate(views):
            azim = math.radians(float(view["azim"]))
            elev = math.radians(float(view["elev"]))
            distance = fit_distance_for_mesh(
                radius,
                camera,
                framing_margin,
                float(view.get("distance_scale", args.camera_distance_scale)),
            )
            camera.location = (
                center.x + distance * math.cos(elev) * math.sin(azim),
                center.y - distance * math.cos(elev) * math.cos(azim),
                center.z + distance * math.sin(elev),
            )
            look_at(camera, center)
            name = f"hdri_{hdri_idx:02d}_view_{view_idx:02d}.png"
            path = out_dir / name
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            render_records.append(
                {
                    "path": str(path),
                    "hdri_idx": hdri_idx,
                    "hdri": hdri.get("name", ""),
                    "view_idx": view_idx,
                    "azim": float(view["azim"]),
                    "elev": float(view["elev"]),
                }
            )

    with open(out_dir / "render_records.json", "w", encoding="utf-8") as f:
        json.dump(render_records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
'''


BLENDER_OBJ_TO_GLB_SCRIPT = r'''
import argparse
from pathlib import Path

import bpy


def parse_args():
    argv = __import__("sys").argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--shade", default="SMOOTH", choices=["SMOOTH", "FLAT"])
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_obj(path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {path}")
    return mesh_objects


def select_meshes(mesh_objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]


def apply_shading(mesh_objects, shade):
    select_meshes(mesh_objects)
    if shade == "SMOOTH":
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()


def main():
    args = parse_args()
    glb_path = Path(args.glb)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    clear_scene()
    mesh_objects = import_obj(args.obj)
    apply_shading(mesh_objects, args.shade)
    bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB", use_active_scene=True)
    if not glb_path.exists():
        raise RuntimeError(f"GLB export failed: {glb_path}")


if __name__ == "__main__":
    main()
'''


@dataclass
class BakeConfig:
    bake_exp: float
    candidate_camera_azims: list
    candidate_camera_elevs: list
    candidate_view_weights: list
    max_selected_view_num: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "LoRA GRPO training for MaterialMVP with Blender-rendered Aesthetic Predictor V2.5 reward. "
            "This is a standalone stage script and does not import previous grpo_stage scripts."
        )
    )
    parser.add_argument("--mesh-path", default=None, help="Input blank mesh path (.obj/.glb).")
    parser.add_argument("--image-path", default=None, help="Reference image path.")
    parser.add_argument("--cases-json", default=None, help="Optional JSON list with {name, mesh, image}.")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--pretrained-model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--pretrained-subdir", default="hunyuan3d-paintpbr-v2-1")
    parser.add_argument("--dino-ckpt-path", default="facebook/dinov2-giant")
    parser.add_argument("--out-dir", default="outputs/grpo_stage4_blender_aesthetic")
    parser.add_argument("--resolution", type=int, default=256, help="MaterialMVP multiview generation resolution.")
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--train-timestep-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--adam-eps", type=float, default=1e-6)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--use-remesh", action="store_true", help="Run the repo remesher before UV wrapping.")
    parser.add_argument("--skip-uvwrap", action="store_true", help="Use the mesh UVs as-is.")
    parser.add_argument("--raster-mode", default="cr", choices=["cr", "nvdiffrast"])
    parser.add_argument("--bake-render-size", type=int, default=2048)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--bake-exp", type=float, default=4.0)
    parser.add_argument("--downsample-saved-textures", action="store_true")
    parser.add_argument(
        "--gen-camera-azims",
        default="0,90,180,270,0,180",
        help="Comma-separated camera azimuths used for MaterialMVP normal/position maps and texture baking.",
    )
    parser.add_argument(
        "--gen-camera-elevs",
        default="0,0,0,0,90,-90",
        help="Comma-separated camera elevations used for MaterialMVP normal/position maps and texture baking.",
    )
    parser.add_argument(
        "--gen-view-weights",
        default="1,0.1,0.5,0.1,0.05,0.05",
        help="Comma-separated bake weights for generation views.",
    )

    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
    parser.add_argument("--render-samples", type=int, default=64)
    parser.add_argument("--render-camera-azims", default="0,60,120,180,240,300")
    parser.add_argument("--render-camera-elevs", default="15,15,15,15,15,15")
    parser.add_argument("--render-camera-distance-scale", type=float, default=2.4)
    parser.add_argument("--render-lens", type=float, default=50.0)
    parser.add_argument("--render-framing-margin", type=float, default=1.25)
    parser.add_argument(
        "--render-mesh-format",
        default="glb",
        choices=["glb", "obj"],
        help="Mesh file used by Blender reward rendering. GLB is converted from the baked OBJ.",
    )
    parser.add_argument(
        "--render-bg-color",
        default="127,127,127",
        help="RGB background color seen by the camera, e.g. 127,127,127.",
    )
    parser.add_argument(
        "--hdri-paths",
        nargs="*",
        default=[],
        help="Environment map paths. If omitted, Blender uses a simple white world plus area lights.",
    )
    parser.add_argument("--hdri-strength", type=float, default=1.0)
    parser.add_argument("--reward-batch-size", type=int, default=8)
    parser.add_argument("--reward-device", default=None)
    parser.add_argument("--reward-dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--aesthetic-encoder-path",
        default="google/siglip-so400m-patch14-384",
        help="SigLIP encoder repo id or local snapshot path used by Aesthetic Predictor V2.5.",
    )
    parser.add_argument(
        "--aesthetic-predictor-path",
        default=None,
        help="Local aesthetic_predictor_v2_5.pth path. Required on fully offline servers if not torch-hub cached.",
    )
    parser.add_argument("--reward-mean-weight", type=float, default=0.7)
    parser.add_argument("--reward-bottom-weight", type=float, default=0.3)
    parser.add_argument("--reward-std-penalty", type=float, default=0.1)
    parser.add_argument("--render-timeout", type=int, default=0, help="Seconds. 0 means no timeout.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def parse_float_list(text):
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one comma-separated value, got: {text}")
    return values


def tensor_to_pil(image_chw):
    array = image_chw.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def pil_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_rgb_pil(path, size=None, bg_color=(255, 255, 255)):
    image = Image.open(path)
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, bg_color)
        bg.paste(image, mask=image.getchannel("A"))
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")
    if size is not None:
        image = image.resize((size, size))
    return image


def resolve_model_path(model_path, subdir):
    local_path = Path(model_path)
    if local_path.exists():
        return str(local_path / subdir) if subdir and (local_path / subdir).exists() else str(local_path)

    import huggingface_hub

    snapshot = huggingface_hub.snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )
    return str(Path(snapshot) / subdir) if subdir else snapshot


def load_case(args):
    if args.cases_json:
        cases_path = Path(args.cases_json)
        with open(cases_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        case = cases[args.case_index]
        root = cases_path.parent
        mesh_path = Path(case["mesh"])
        image_path = Path(case["image"])
        if not mesh_path.is_absolute():
            mesh_path = root / mesh_path
        if not image_path.is_absolute():
            image_path = root / image_path
        return {
            "name": case.get("name", f"case_{args.case_index:04d}"),
            "mesh_path": str(mesh_path),
            "image_path": str(image_path),
        }

    if not args.mesh_path or not args.image_path:
        raise ValueError("Provide either --mesh-path/--image-path or --cases-json.")
    return {
        "name": Path(args.mesh_path).stem,
        "mesh_path": args.mesh_path,
        "image_path": args.image_path,
    }


def load_original_mvp_pipeline_for_ddim(args):
    dtype = resolve_dtype(args.dtype)
    model_path = resolve_model_path(args.pretrained_model_path, args.pretrained_subdir)
    pipe = DiffusionPipeline.from_pretrained(model_path, custom_pipeline="materialmvp", torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    pipe.eval()
    pipe.to(args.device)

    dino = None
    if getattr(pipe.unet, "use_dino", False):
        from materialmvp.modules import Dino_v2

        dino = Dino_v2(args.dino_ckpt_path).to(device=args.device, dtype=dtype)
        dino.eval().requires_grad_(False)
    return pipe, dino


def get_material_latent_channels(pipe):
    latent_channels = getattr(pipe.vae.config, "latent_channels", None)
    if latent_channels is None:
        latent_channels = getattr(pipe.unet.config, "out_channels", None)
    if latent_channels is None:
        latent_channels = 4
    return int(latent_channels)


def print_channel_config(pipe):
    print("UNet config in_channels:", getattr(pipe.unet.config, "in_channels", None))
    print("UNet config out_channels:", getattr(pipe.unet.config, "out_channels", None))
    print("VAE latent_channels:", getattr(pipe.vae.config, "latent_channels", None))
    print("Material latent channels used for rollout:", get_material_latent_channels(pipe))


def pil_batch_to_tensor(batch_images, device, dtype):
    batches = []
    for images in batch_images:
        views = []
        for pil_img in images:
            image = to_rgb_image(pil_img)
            tensor = pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
            views.append(tensor)
        batches.append(torch.cat(views, dim=0).unsqueeze(0))
    return torch.cat(batches, dim=0)


@torch.no_grad()
def prepare_materialmvp_conditions(pipe, dino, cond_pil, normal_pils, position_pils, guidance_scale):
    pipe.prepare()
    device = pipe.vae.device
    dtype = pipe.unet.dtype

    image = to_rgb_image(cond_pil)
    image_vae = torch.tensor(np.array(image) / 255.0)
    image_vae = image_vae.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
    batch_size = image_vae.shape[0]

    cached_condition = {
        "num_in_batch": len(normal_pils),
        "images_normal": pil_batch_to_tensor([normal_pils], device, dtype),
        "images_position": pil_batch_to_tensor([position_pils], device, dtype),
    }

    if getattr(pipe.unet, "use_ra", False):
        cached_condition["ref_latents"] = pipe.encode_images(image_vae)

    cached_condition["embeds_normal"] = pipe.encode_images(cached_condition["images_normal"])
    cached_condition["position_maps"] = cached_condition["images_position"]
    cached_condition["embeds_position"] = pipe.encode_images(cached_condition["images_position"])

    if getattr(pipe.unet, "use_dino", False):
        if dino is None:
            raise ValueError("Pipeline UNet uses DINO but no DINO model was loaded.")
        cached_condition["dino_hidden_states"] = dino(cond_pil)

    if getattr(pipe.unet, "use_learned_text_clip", False):
        tokens = []
        for token in pipe.unet.pbr_setting:
            tokens.append(getattr(pipe.unet, f"learned_text_clip_{token}").unsqueeze(0).repeat(batch_size, 1, 1))
        prompt_embeds = torch.stack(tokens, dim=1)
        negative_prompt_embeds = torch.stack(tokens, dim=1)
    else:
        prompt_embeds, _ = pipe.encode_prompt(["high quality"], device, 1, False)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    if guidance_scale > 1:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds])
        if "ref_latents" in cached_condition:
            cached_condition["ref_latents"] = cached_condition["ref_latents"].repeat(
                3, *([1] * (cached_condition["ref_latents"].dim() - 1))
            )
            cached_condition["ref_scale"] = torch.as_tensor([0.0, 1.0, 1.0]).to(cached_condition["ref_latents"])
        if "dino_hidden_states" in cached_condition:
            zero_states = torch.zeros_like(cached_condition["dino_hidden_states"])
            cached_condition["dino_hidden_states"] = torch.cat(
                [zero_states, zero_states, cached_condition["dino_hidden_states"]]
            )
        for key in ["embeds_normal", "embeds_position", "position_maps"]:
            cached_condition[key] = cached_condition[key].repeat(3, *([1] * (cached_condition[key].dim() - 1)))
    return prompt_embeds, cached_condition


def apply_materialmvp_guidance(noise_pred, num_view, n_pbr, guidance_scale, camera_azims=None):
    noise_pred_uncond, noise_pred_ref, noise_pred_full = noise_pred.chunk(3)
    camera_azims = camera_azims or [0] * num_view

    def cam_mapping(azim):
        if 0 <= azim < 90:
            return float(azim) / 90.0 + 1.0
        if 90 <= azim < 330:
            return 2.0
        return -float(azim) / 90.0 + 5.0

    view_scale_tensor = (
        torch.from_numpy(np.asarray([cam_mapping(azim) for azim in camera_azims]))
        .unsqueeze(0)
        .repeat(n_pbr, 1)
        .view(-1)
        .to(noise_pred_uncond)[:, None, None, None]
    )
    guided = noise_pred_uncond + guidance_scale * view_scale_tensor * (noise_pred_ref - noise_pred_uncond)
    guided = guided + guidance_scale * view_scale_tensor * (noise_pred_full - noise_pred_ref)
    return guided


@torch.no_grad()
def rollout_with_logprob(pipe, dino, cond_pil, normal_pils, position_pils, camera_azims, args, seed):
    if args.eta <= 0:
        raise ValueError("--eta must be > 0 for stochastic DDIM log-prob rollouts.")

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    prompt_embeds, cached_condition = prepare_materialmvp_conditions(
        pipe,
        dino,
        cond_pil,
        normal_pils,
        position_pils,
        args.guidance_scale,
    )
    rollout_condition = copy.copy(cached_condition)
    rollout_condition["cache"] = {}

    device = pipe._execution_device
    n_pbr = len(pipe.unet.pbr_setting)
    num_view = len(normal_pils)
    num_channels_latents = get_material_latent_channels(pipe)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latents = pipe.prepare_latents(
        num_view * n_pbr,
        num_channels_latents,
        args.resolution,
        args.resolution,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )

    all_latents = []
    all_next_latents = []
    all_log_probs = []
    all_noise_pred_norms = []

    for timestep in timesteps:
        latents_before = latents
        latent_grid = rearrange(latents, "(b n_pbr n) c h w -> b n_pbr n c h w", b=1, n_pbr=n_pbr, n=num_view)
        latent_model_input = latent_grid.repeat(3, 1, 1, 1, 1, 1) if args.guidance_scale > 1 else latent_grid
        latent_model_input = rearrange(latent_model_input, "b n_pbr n c h w -> (b n_pbr n) c h w")
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        latent_model_input = rearrange(
            latent_model_input,
            "(b n_pbr n) c h w -> b n_pbr n c h w",
            n=num_view,
            n_pbr=n_pbr,
        )

        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
            **rollout_condition,
        )[0]
        if args.guidance_scale > 1:
            noise_pred = apply_materialmvp_guidance(noise_pred, num_view, n_pbr, args.guidance_scale, camera_azims)

        latents, log_prob = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            timestep,
            latents_before[:, :num_channels_latents],
            eta=args.eta,
            generator=generator,
        )

        all_latents.append(latents_before.detach().cpu())
        all_next_latents.append(latents.detach().cpu())
        all_log_probs.append(log_prob.detach().float().cpu())
        all_noise_pred_norms.append(float(noise_pred.detach().float().norm().cpu()))

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
    image = (image * 0.5 + 0.5).clamp(0, 1)
    image_pils = [tensor_to_pil(image_i) for image_i in image]

    generated = {
        "albedo": image_pils[:num_view],
        "mr": image_pils[num_view : 2 * num_view],
    }
    trajectory = {
        "timesteps": timesteps.detach().cpu(),
        "latents": torch.stack(all_latents, dim=0),
        "next_latents": torch.stack(all_next_latents, dim=0),
        "old_log_probs": torch.stack(all_log_probs, dim=0),
        "noise_pred_norms": all_noise_pred_norms,
    }
    return generated, trajectory


def prepare_training_condition(pipe, dino, cond_pil, normal_pils, position_pils, args):
    prompt_embeds, cached_condition = prepare_materialmvp_conditions(
        pipe,
        dino,
        cond_pil,
        normal_pils,
        position_pils,
        args.guidance_scale,
    )
    if getattr(pipe.unet, "use_learned_text_clip", False):
        batch_size = 1
        tokens = []
        for token in pipe.unet.pbr_setting:
            tokens.append(getattr(pipe.unet, f"learned_text_clip_{token}").unsqueeze(0).repeat(batch_size, 1, 1))
        prompt_embeds_grad = torch.stack(tokens, dim=1)
        negative_prompt_embeds_grad = torch.stack(tokens, dim=1)
        if args.guidance_scale > 1:
            prompt_embeds = torch.cat([negative_prompt_embeds_grad, prompt_embeds_grad, prompt_embeds_grad])
        else:
            prompt_embeds = prompt_embeds_grad
    train_condition = copy.copy(cached_condition)
    train_condition["cache"] = {}
    return prompt_embeds, train_condition


def grpo_loss_from_log_probs(new_log_probs, old_log_probs, advantage, args):
    old_log_probs = old_log_probs.to(new_log_probs.device, dtype=new_log_probs.dtype)
    advantage = torch.as_tensor(advantage, device=new_log_probs.device, dtype=new_log_probs.dtype)
    advantage = advantage.clamp(-args.adv_clip_max, args.adv_clip_max)

    ratio = torch.exp((new_log_probs - old_log_probs).clamp(-20, 20))
    unclipped = -advantage * ratio
    clipped = -advantage * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
    loss = torch.maximum(unclipped, clipped).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    approx_kl = 0.5 * (new_log_probs - old_log_probs).pow(2).mean()
    clipfrac = (torch.abs(ratio - 1.0) > args.clip_range).float().mean()
    return loss, {
        "ratio_mean": float(ratio.detach().mean().cpu()),
        "ratio_std": float(ratio.detach().std(unbiased=False).cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "clipfrac": float(clipfrac.detach().cpu()),
    }


def backward_grpo_for_trajectory(pipe, dino, sample, trajectory, advantage, args, loss_scale=1.0):
    device = pipe._execution_device
    total_steps = int(trajectory["timesteps"].numel())
    train_steps = max(1, int(total_steps * args.train_timestep_fraction))
    step_indices = list(range(train_steps))

    losses = []
    ratio_means = []
    ratio_stds = []
    approx_kls = []
    clipfracs = []
    old_log_probs_all = trajectory["old_log_probs"]

    for step_idx in step_indices:
        prompt_embeds, cached_condition = prepare_training_condition(
            pipe,
            dino,
            sample["cond_pil"],
            sample["normal_pils"],
            sample["position_pils"],
            args,
        )
        dtype = prompt_embeds.dtype
        n_pbr = len(pipe.unet.pbr_setting)
        num_view = len(sample["normal_pils"])
        num_channels_latents = get_material_latent_channels(pipe)

        timestep = trajectory["timesteps"][step_idx].to(device)
        latents_t = trajectory["latents"][step_idx].to(device=device, dtype=dtype)
        next_latents_t = trajectory["next_latents"][step_idx].to(device=device, dtype=dtype)

        latent_grid = rearrange(
            latents_t,
            "(b n_pbr n) c h w -> b n_pbr n c h w",
            b=1,
            n_pbr=n_pbr,
            n=num_view,
        )
        latent_model_input = latent_grid.repeat(3, 1, 1, 1, 1, 1) if args.guidance_scale > 1 else latent_grid
        latent_model_input = rearrange(latent_model_input, "b n_pbr n c h w -> (b n_pbr n) c h w")
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        latent_model_input = rearrange(
            latent_model_input,
            "(b n_pbr n) c h w -> b n_pbr n c h w",
            n=num_view,
            n_pbr=n_pbr,
        )

        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
            **cached_condition,
        )[0]
        if args.guidance_scale > 1:
            noise_pred = apply_materialmvp_guidance(
                noise_pred,
                num_view,
                n_pbr,
                args.guidance_scale,
                sample["camera_azims"],
            )

        _prev_sample, new_log_prob = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            timestep,
            latents_t[:, :num_channels_latents],
            eta=args.eta,
            prev_sample=next_latents_t[:, :num_channels_latents],
        )

        old_log_prob = old_log_probs_all[step_idx].to(new_log_prob.device, dtype=new_log_prob.dtype)
        loss, info = grpo_loss_from_log_probs(new_log_prob, old_log_prob, advantage, args)
        (loss * loss_scale / len(step_indices)).backward()

        losses.append(float(loss.detach().cpu()))
        ratio_means.append(info["ratio_mean"])
        ratio_stds.append(info["ratio_std"])
        approx_kls.append(info["approx_kl"])
        clipfracs.append(info["clipfrac"])
        del prompt_embeds, cached_condition, latent_model_input, noise_pred, new_log_prob, loss

    def mean(values):
        return float(sum(values) / max(len(values), 1))

    return {
        "trained_timestep_indices": step_indices,
        "loss": mean(losses),
        "ratio_mean": mean(ratio_means),
        "ratio_std": mean(ratio_stds),
        "approx_kl": mean(approx_kls),
        "clipfrac": mean(clipfracs),
    }


def setup_trainable_lora(pipe, args):
    report = inject_lora_into_attention(
        pipe.unet,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_suffixes=(
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "to_q_mr",
            "to_k_mr",
            "to_v_mr",
            "to_out_mr.0",
        ),
        include_keywords=("attn1", "attn2", "attn_refview"),
        exclude_keywords=("unet_dual", "attn_multiview", "attn_dino"),
        extra_trainable_keywords=(
            "learned_text_clip_albedo",
            "learned_text_clip_mr",
            "learned_text_clip_ref",
        ),
        freeze_first=True,
    )
    print(
        "Stage-4 LoRA enabled: "
        f"modules={report.module_count}, trainable={report.trainable_params}, total={report.total_params}, "
        f"ratio={report.trainable_params / max(report.total_params, 1):.6f}"
    )
    return report


def trainable_parameters(pipe):
    params = [param for param in pipe.unet.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found after LoRA setup.")
    return params


def trainable_named_parameters(pipe):
    return [(name, param) for name, param in pipe.unet.named_parameters() if param.requires_grad]


def sanitize_gradients(pipe):
    bad = []
    for name, param in trainable_named_parameters(pipe):
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(name)
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return bad


def finite_parameter_report(pipe):
    bad = []
    abs_max = 0.0
    for name, param in trainable_named_parameters(pipe):
        data = param.detach()
        if not torch.isfinite(data).all():
            bad.append(name)
            continue
        if data.numel() > 0:
            abs_max = max(abs_max, float(data.float().abs().max().cpu()))
    return bad, abs_max


def save_stage4_checkpoint(path, pipe, lora_report, args, update_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    lora_config = {
        **lora_report.config,
        "extra_trainable_keywords": [
            "learned_text_clip_albedo",
            "learned_text_clip_mr",
            "learned_text_clip_ref",
        ],
        "stage": "grpo_stage4_blender_aesthetic",
        "update_idx": update_idx,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "reward": "aesthetic_predictor_v2_5_blender_render",
    }
    save_lora_checkpoint(str(path), pipe.unet, lora_config)


class AestheticV25Reward:
    def __init__(
        self,
        device,
        dtype_name,
        batch_size,
        mean_weight,
        bottom_weight,
        std_penalty,
        encoder_name_or_path,
        predictor_name_or_path,
    ):
        try:
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        except ImportError as exc:
            raise ImportError(
                "Aesthetic Predictor V2.5 is not installed. Install it with: "
                "pip install aesthetic-predictor-v2-5"
            ) from exc

        self.device = device
        self.dtype = resolve_dtype(dtype_name)
        self.batch_size = batch_size
        self.mean_weight = mean_weight
        self.bottom_weight = bottom_weight
        self.std_penalty = std_penalty
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            predictor_name_or_path=predictor_name_or_path,
            encoder_model_name=encoder_name_or_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = self.model.to(device=device, dtype=self.dtype).eval()

    @torch.no_grad()
    def score_images(self, image_paths):
        scores = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            pixel_values = self.preprocessor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            batch_scores = self.model(pixel_values).logits.squeeze(-1).float().detach().cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend([float(score) for score in batch_scores])
        return scores

    def score_group(self, image_paths):
        scores = self.score_images(image_paths)
        if not scores:
            raise ValueError("No rendered images were provided to the reward model.")
        scores_np = np.asarray(scores, dtype=np.float32)
        bottom_k = max(1, int(math.ceil(len(scores_np) * 0.2)))
        bottom_mean = float(np.sort(scores_np)[:bottom_k].mean())
        mean_score = float(scores_np.mean())
        std_score = float(scores_np.std())
        reward = self.mean_weight * mean_score + self.bottom_weight * bottom_mean - self.std_penalty * std_score
        return {
            "reward": float(reward),
            "aesthetic_mean": mean_score,
            "aesthetic_std": std_score,
            "aesthetic_min": float(scores_np.min()),
            "aesthetic_max": float(scores_np.max()),
            "aesthetic_bottom20": bottom_mean,
            "num_rendered_images": len(scores),
            "scores": scores,
        }


def write_blender_script(out_dir):
    script_path = out_dir / "_stage4_blender_render.py"
    script_path.write_text(BLENDER_RENDER_SCRIPT, encoding="utf-8")
    return script_path


def write_blender_convert_script(out_dir):
    script_path = out_dir / "_stage4_obj_to_glb.py"
    script_path.write_text(BLENDER_OBJ_TO_GLB_SCRIPT, encoding="utf-8")
    return script_path


def build_render_plan(args):
    azims = parse_float_list(args.render_camera_azims)
    elevs = parse_float_list(args.render_camera_elevs)
    bg_color = parse_float_list(args.render_bg_color)
    if len(azims) != len(elevs):
        raise ValueError("--render-camera-azims and --render-camera-elevs must have the same length.")
    if len(bg_color) != 3:
        raise ValueError("--render-bg-color must contain exactly 3 comma-separated RGB values.")
    bg_color = [int(max(0, min(255, round(value)))) for value in bg_color]
    views = [
        {
            "azim": azim,
            "elev": elev,
            "distance_scale": args.render_camera_distance_scale,
        }
        for azim, elev in zip(azims, elevs)
    ]

    hdris = []
    for idx, path in enumerate(args.hdri_paths):
        resolved = str(Path(path).expanduser().resolve())
        hdris.append({"name": Path(path).stem or f"hdri_{idx}", "path": resolved, "strength": args.hdri_strength})
    return {
        "views": views,
        "hdris": hdris,
        "world_strength": args.hdri_strength,
        "lens": args.render_lens,
        "framing_margin": args.render_framing_margin,
        "background_color": bg_color,
    }


def render_candidate_with_blender(mesh_path, out_dir, blender_script, render_plan, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "render_plan.json"
    plan_path.write_text(json.dumps(render_plan, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd = [
        args.blender_bin,
        "-b",
        "--python",
        str(blender_script),
        "--",
        "--mesh",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        "--plan",
        str(plan_path),
        "--resolution",
        str(args.render_resolution),
        "--engine",
        args.render_engine,
        "--samples",
        str(args.render_samples),
        "--camera-distance-scale",
        str(args.render_camera_distance_scale),
    ]
    timeout = None if args.render_timeout <= 0 else args.render_timeout
    subprocess.run(cmd, check=True, timeout=timeout)
    render_paths = sorted(str(path) for path in out_dir.glob("*.png"))
    if not render_paths:
        raise RuntimeError(f"Blender rendered no PNG files in {out_dir}")
    return render_paths


def convert_obj_to_glb_with_blender(obj_path, blender_script, args):
    obj_path = Path(obj_path)
    glb_path = obj_path.with_suffix(".glb")
    cmd = [
        args.blender_bin,
        "-b",
        "--python",
        str(blender_script),
        "--",
        "--obj",
        str(obj_path),
        "--glb",
        str(glb_path),
        "--shade",
        "SMOOTH",
    ]
    timeout = None if args.render_timeout <= 0 else args.render_timeout
    subprocess.run(cmd, check=True, timeout=timeout)
    if not glb_path.exists():
        raise RuntimeError(f"Expected GLB was not created: {glb_path}")
    return glb_path


def choose_render_mesh_path(obj_path, blender_convert_script, args):
    if args.render_mesh_format == "obj":
        return Path(obj_path)
    return convert_obj_to_glb_with_blender(obj_path, blender_convert_script, args)


def save_contact_sheet(generated, path):
    rows = [generated["albedo"], generated["mr"]]
    cell_w, cell_h = rows[0][0].size
    sheet = Image.new("RGB", (cell_w * len(rows[0]), cell_h * len(rows)), (127, 127, 127))
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            sheet.paste(image.convert("RGB"), (col_idx * cell_w, row_idx * cell_h))
    sheet.save(path)


def prepare_mesh_sample(case, args, out_dir):
    import trimesh
    from DifferentiableRenderer.MeshRender import MeshRender
    from utils.pipeline_utils import ViewProcessor
    from utils.simplify_mesh_utils import remesh_mesh
    from utils.uvwrap_utils import mesh_uv_wrap

    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(case["mesh_path"]).expanduser().resolve()
    image_path = Path(case["image_path"]).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    working_mesh_path = mesh_path
    if args.use_remesh:
        remesh_path = out_dir / "prepared_mesh_remesh.obj"
        remesh_mesh(str(mesh_path), str(remesh_path))
        working_mesh_path = remesh_path

    mesh = trimesh.load(str(working_mesh_path), force="mesh")
    if not args.skip_uvwrap:
        mesh = mesh_uv_wrap(mesh)

    gen_azims = parse_float_list(args.gen_camera_azims)
    gen_elevs = parse_float_list(args.gen_camera_elevs)
    gen_weights = parse_float_list(args.gen_view_weights)
    if not (len(gen_azims) == len(gen_elevs) == len(gen_weights)):
        raise ValueError("--gen-camera-azims, --gen-camera-elevs and --gen-view-weights must have the same length.")

    bake_cfg = BakeConfig(
        bake_exp=args.bake_exp,
        candidate_camera_azims=gen_azims,
        candidate_camera_elevs=gen_elevs,
        candidate_view_weights=gen_weights,
        max_selected_view_num=len(gen_azims),
    )

    render = MeshRender(
        default_resolution=args.bake_render_size,
        texture_size=args.texture_size,
        bake_mode="back_sample",
        raster_mode=args.raster_mode,
    )
    view_processor = ViewProcessor(bake_cfg, render)
    render.load_mesh(mesh=mesh)

    normal_pils = view_processor.render_normal_multiview(gen_elevs, gen_azims, use_abs_coor=True)
    position_pils = view_processor.render_position_multiview(gen_elevs, gen_azims)
    cond_pil = load_rgb_pil(image_path, args.resolution)

    cond_pil.save(out_dir / "reference.png")
    save_contact_sheet({"albedo": normal_pils, "mr": position_pils}, out_dir / "mesh_normal_position_condition.png")

    return {
        "name": case["name"],
        "mesh_path": str(mesh_path),
        "image_path": str(image_path),
        "cond_pil": cond_pil,
        "normal_pils": [image.resize((args.resolution, args.resolution)) for image in normal_pils],
        "position_pils": [image.resize((args.resolution, args.resolution)) for image in position_pils],
        "camera_azims": gen_azims,
        "camera_elevs": gen_elevs,
        "view_weights": gen_weights,
        "render": render,
        "view_processor": view_processor,
    }


def bake_generated_to_mesh(sample, generated, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    render = sample["render"]
    view_processor = sample["view_processor"]
    camera_elevs = sample["camera_elevs"]
    camera_azims = sample["camera_azims"]
    view_weights = sample["view_weights"]

    albedo_views = [image.resize((args.bake_render_size, args.bake_render_size)) for image in generated["albedo"]]
    mr_views = [image.resize((args.bake_render_size, args.bake_render_size)) for image in generated["mr"]]
    texture, mask = view_processor.bake_from_multiview(albedo_views, camera_elevs, camera_azims, view_weights)
    texture_mr, mask_mr = view_processor.bake_from_multiview(mr_views, camera_elevs, camera_azims, view_weights)

    mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    texture = view_processor.texture_inpaint(texture, mask_np)
    texture_mr = view_processor.texture_inpaint(texture_mr, mask_mr_np)

    render.set_texture(texture, force_set=True)
    render.set_texture_mr(texture_mr, force_set=True)

    obj_path = out_dir / "candidate_textured.obj"
    render.save_mesh(str(obj_path), downsample=args.downsample_saved_textures)
    return obj_path


def trajectory_summary(trajectory):
    log_probs = trajectory["old_log_probs"].float()
    timesteps = trajectory["timesteps"].long()
    return {
        "num_steps": int(timesteps.numel()),
        "timesteps": [int(item) for item in timesteps.tolist()],
        "latents_shape": list(trajectory["latents"].shape),
        "old_log_probs_mean": float(log_probs.mean()),
        "old_log_probs_std": float(log_probs.std(unbiased=False)),
        "old_log_probs_min": float(log_probs.min()),
        "old_log_probs_max": float(log_probs.max()),
        "noise_pred_norms": trajectory["noise_pred_norms"],
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.eta <= 0:
        raise ValueError("--eta must be > 0.")
    if not (0 < args.train_timestep_fraction <= 1):
        raise ValueError("--train-timestep-fraction must be in (0, 1].")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blender_script = write_blender_script(out_dir)
    blender_convert_script = write_blender_convert_script(out_dir)
    render_plan = build_render_plan(args)
    case = load_case(args)

    pipe, dino = load_original_mvp_pipeline_for_ddim(args)
    print_channel_config(pipe)
    lora_report = setup_trainable_lora(pipe, args)
    optimizer = torch.optim.AdamW(trainable_parameters(pipe), lr=args.learning_rate, eps=args.adam_eps)

    reward_device = args.reward_device or args.device
    reward_model = AestheticV25Reward(
        device=reward_device,
        dtype_name=args.reward_dtype,
        batch_size=args.reward_batch_size,
        mean_weight=args.reward_mean_weight,
        bottom_weight=args.reward_bottom_weight,
        std_penalty=args.reward_std_penalty,
        encoder_name_or_path=args.aesthetic_encoder_path,
        predictor_name_or_path=args.aesthetic_predictor_path,
    )

    sample = prepare_mesh_sample(case, args, out_dir / "input")
    update_records = []

    for update_idx in range(args.max_updates):
        sample_out_dir = out_dir / f"update_{update_idx:04d}_{sample['name']}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)

        pipe.unet.eval()
        group_payloads = []
        for group_idx in range(args.group_size):
            group_dir = sample_out_dir / f"group_{group_idx:02d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            seed = args.seed + update_idx * 1000 + group_idx
            generated, trajectory = rollout_with_logprob(
                pipe,
                dino,
                sample["cond_pil"],
                sample["normal_pils"],
                sample["position_pils"],
                sample["camera_azims"],
                args,
                seed,
            )
            save_contact_sheet(generated, group_dir / "multiview_albedo_mr.png")
            textured_obj_path = bake_generated_to_mesh(sample, generated, group_dir / "mesh", args)
            textured_mesh_path = choose_render_mesh_path(textured_obj_path, blender_convert_script, args)
            render_paths = render_candidate_with_blender(
                textured_mesh_path,
                group_dir / "renders",
                blender_script,
                render_plan,
                args,
            )
            reward_info = reward_model.score_group(render_paths)
            metrics_path = group_dir / "reward.json"
            reward_info_with_paths = {**reward_info, "render_paths": render_paths, "mesh_path": str(textured_mesh_path)}
            metrics_path.write_text(json.dumps(reward_info_with_paths, indent=2, ensure_ascii=False), encoding="utf-8")
            group_payloads.append(
                {
                    "group_idx": group_idx,
                    "seed": seed,
                    "reward_info": reward_info,
                    "trajectory": trajectory,
                    "mesh_path": str(textured_mesh_path),
                    "render_paths": render_paths,
                }
            )
            print(
                f"update={update_idx:04d} group={group_idx:02d} "
                f"reward={reward_info['reward']:.4f} aesthetic_mean={reward_info['aesthetic_mean']:.4f} "
                f"renders={len(render_paths)}"
            )

        rewards = torch.tensor([item["reward_info"]["reward"] for item in group_payloads], dtype=torch.float32)
        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False)
        advantages = (rewards - reward_mean) / (reward_std + 1e-8)

        pipe.unet.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss_value = 0.0
        group_records = []

        for payload, advantage in zip(group_payloads, advantages):
            info = backward_grpo_for_trajectory(
                pipe,
                dino,
                sample,
                payload["trajectory"],
                float(advantage),
                args,
                loss_scale=1.0 / len(group_payloads),
            )
            total_loss_value += info["loss"]
            group_records.append(
                {
                    "group_idx": payload["group_idx"],
                    "seed": payload["seed"],
                    "advantage": float(advantage),
                    "mesh_path": payload["mesh_path"],
                    "num_rendered_images": len(payload["render_paths"]),
                    "trajectory": trajectory_summary(payload["trajectory"]),
                    **payload["reward_info"],
                    **info,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        bad_grads = sanitize_gradients(pipe)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(pipe), args.max_grad_norm, error_if_nonfinite=False)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            print("Warning: non-finite grad norm; skipping optimizer step for this update.")
            optimizer.zero_grad(set_to_none=True)
            grad_norm = torch.as_tensor(float("nan"))
            bad_params, param_abs_max = finite_parameter_report(pipe)
        else:
            optimizer.step()
            bad_params, param_abs_max = finite_parameter_report(pipe)
            if bad_params:
                print("Warning: non-finite trainable parameters after optimizer step:")
                for name in bad_params[:20]:
                    print(f"  {name}")
                raise RuntimeError("Non-finite trainable parameters detected after optimizer step.")

        total_loss_value /= max(len(group_payloads), 1)
        record = {
            "update": update_idx,
            "case": case,
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "total_loss": total_loss_value,
            "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            "bad_grad_count": len(bad_grads),
            "bad_param_count": len(bad_params),
            "trainable_param_abs_max": param_abs_max,
            "groups": group_records,
        }
        update_records.append(record)
        (sample_out_dir / "stage4_metrics.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"update={update_idx:04d} loss={record['total_loss']:.6f} "
            f"reward_mean={record['reward_mean']:.4f} reward_std={record['reward_std']:.4f} "
            f"grad_norm={record['grad_norm']:.6f}"
        )

        if args.save_every > 0 and (update_idx + 1) % args.save_every == 0:
            ckpt_path = out_dir / "checkpoints" / f"lora_grpo_aesthetic_step_{update_idx + 1:06d}.pt"
            save_stage4_checkpoint(ckpt_path, pipe, lora_report, args, update_idx + 1)
            print(f"Saved Stage-4 LoRA checkpoint: {ckpt_path}")

    summary = {
        "stage": "grpo_stage4_blender_aesthetic_lora",
        "updates_model": True,
        "trainable": "fresh_lora_plus_learned_text_tokens",
        "case": case,
        "pretrained_model_path": args.pretrained_model_path,
        "pretrained_subdir": args.pretrained_subdir,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "eta": args.eta,
        "guidance_scale": args.guidance_scale,
        "group_size": args.group_size,
        "max_updates": args.max_updates,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "render_plan": render_plan,
        "render_resolution": args.render_resolution,
        "render_mesh_format": args.render_mesh_format,
        "reward_aggregation": {
            "mean_weight": args.reward_mean_weight,
            "bottom_weight": args.reward_bottom_weight,
            "std_penalty": args.reward_std_penalty,
        },
        "lora": lora_report.config,
        "records": update_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    final_path = out_dir / "checkpoints" / "lora_grpo_aesthetic_last.pt"
    save_stage4_checkpoint(final_path, pipe, lora_report, args, args.max_updates)
    print(f"Saved final Stage-4 LoRA checkpoint: {final_path}")
    print(f"Stage-4 summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
