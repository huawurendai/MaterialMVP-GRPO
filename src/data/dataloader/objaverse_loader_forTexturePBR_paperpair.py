import glob
import json
import os
import random
from pathlib import Path

import torch

from .loader_util import BaseDataset


class TextureDatasetPaperPair(BaseDataset):
    def __init__(
        self,
        json_path,
        num_view=6,
        image_size=512,
        point_light_prob=0.4,
        azimuth_view_count=24,
        enable_augmentation=False,
    ):
        self.data = []
        self.num_view = num_view
        self.image_size = image_size
        self.point_light_prob = point_light_prob
        self.azimuth_view_count = azimuth_view_count
        self.enable_augmentation = enable_augmentation
        self._cond_cache = {}
        self._tex_cache = {}

        if isinstance(json_path, str):
            json_path = [json_path]
        for jp in json_path:
            jp_path = Path(jp)
            with open(jp_path) as f:
                entries = json.load(f)
            for entry in entries:
                entry_path = Path(entry)
                if not entry_path.is_absolute():
                    entry_path = jp_path.parent / entry_path
                self.data.append(str(entry_path))
        print("============= length of dataset %d =============" % len(self.data))

    def _load_cond_entries(self, sample_dir):
        if sample_dir in self._cond_cache:
            return self._cond_cache[sample_dir]

        cond_dir = Path(sample_dir) / "render_cond"
        transforms_path = cond_dir / "transforms.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"Missing condition transforms: {transforms_path}")

        with open(transforms_path) as f:
            transforms = json.load(f)

        entries = []
        for frame in transforms.get("frames", []):
            path = cond_dir / frame["file_path"]
            if not path.exists():
                continue
            lighting_type = frame.get("lighting_type")
            if lighting_type is None:
                lighting_type = "PL" if "_light_PL" in path.name else "HDR"
            entries.append(
                {
                    "path": str(path),
                    "azimuth_index": int(frame.get("azimuth_index", 0)),
                    "elevation_index": int(frame.get("elevation_index", -1)),
                    "lighting_type": lighting_type,
                }
            )

        if not entries:
            raise ValueError(f"No condition entries found in {cond_dir}")
        self._cond_cache[sample_dir] = entries
        return entries

    def _is_neighbor_azimuth(self, a, b):
        diff = abs(int(a) - int(b)) % self.azimuth_view_count
        diff = min(diff, self.azimuth_view_count - diff)
        return diff <= 1

    def _sample_ref_pair(self, sample_dir):
        entries = self._load_cond_entries(sample_dir)
        point_entries = [entry for entry in entries if entry["lighting_type"] == "PL"]
        if point_entries and random.random() < self.point_light_prob:
            first = random.choice(point_entries)
        else:
            first = random.choice(entries)

        candidates = [
            entry
            for entry in entries
            if entry["path"] != first["path"] and self._is_neighbor_azimuth(entry["azimuth_index"], first["azimuth_index"])
        ]
        if not candidates:
            candidates = [entry for entry in entries if entry["path"] != first["path"]]
        if not candidates:
            candidates = [first]
        second = random.choice(candidates)
        return [first["path"], second["path"]]

    def _load_tex_entries(self, sample_dir):
        if sample_dir in self._tex_cache:
            return self._tex_cache[sample_dir]

        render_tex = Path(sample_dir) / "render_tex"
        transforms_path = render_tex / "transforms.json"
        entries = []

        if transforms_path.exists():
            with open(transforms_path) as f:
                transforms = json.load(f)
            for frame in transforms.get("frames", []):
                stem = Path(frame["file_path"]).stem
                albedo = render_tex / f"{stem}_albedo.png"
                if albedo.exists():
                    entries.append(
                        {
                            "path": str(albedo),
                            "azimuth_index": int(frame.get("azimuth_index", len(entries))),
                            "elevation_index": int(frame.get("elevation_index", -1)),
                        }
                    )

        if not entries:
            available_views = []
            for ext in ["*_albedo.png", "*_albedo.jpg", "*_albedo.jpeg"]:
                available_views.extend(glob.glob(os.path.join(sample_dir, "render_tex", ext)))
            entries = [
                {"path": path, "azimuth_index": idx, "elevation_index": -1}
                for idx, path in enumerate(sorted(available_views))
            ]

        if not entries:
            raise ValueError(f"No render_tex albedo images found in {render_tex}")
        self._tex_cache[sample_dir] = entries
        return entries

    def _sample_tex_views(self, sample_dir):
        entries = self._load_tex_entries(sample_dir)
        by_elevation = {}
        for entry in entries:
            by_elevation.setdefault(entry["elevation_index"], []).append(entry)

        eligible_groups = [group for group in by_elevation.values() if len(group) >= self.num_view]
        if eligible_groups:
            group = sorted(random.choice(eligible_groups), key=lambda item: item["azimuth_index"])
            if len(group) >= self.num_view and len(group) % self.num_view == 0:
                stride = len(group) // self.num_view
                start = random.randrange(stride)
                return [group[(start + i * stride) % len(group)]["path"] for i in range(self.num_view)]
            return [entry["path"] for entry in random.sample(group, self.num_view)]

        paths = [entry["path"] for entry in entries]
        if len(paths) < self.num_view:
            print(f"Warning: Only {len(paths)} views available, but {self.num_view} requested. Using all available views.")
            return paths
        return random.sample(paths, self.num_view)

    def _maybe_augment(self, image, bg_color):
        if not self.enable_augmentation:
            return image
        return self.augment_image(image, bg_color)

    def __getitem__(self, index):
        images_ref = []
        images_albedo = []
        images_mr = []
        images_normal = []
        images_position = []
        bg_white = [1.0, 1.0, 1.0]
        bg_black = [0.0, 0.0, 0.0]
        bg_gray = [127 / 255.0, 127 / 255.0, 127 / 255.0]
        sample_dir = self.data[index]

        images_ref_paths = self._sample_ref_pair(sample_dir)
        bg_c_record = None
        for i, image_ref in enumerate(images_ref_paths):
            if random.random() < 0.6:
                bg_c = bg_gray
            else:
                bg_c = bg_black if random.random() < 0.5 else bg_white
            if i == 0:
                bg_c_record = bg_c
            image, _alpha = self.load_image(image_ref, bg_c_record)
            images_ref.append(self._maybe_augment(image, bg_c_record).float())

        for image_gen in self._sample_tex_views(sample_dir):
            images_albedo.append(self._maybe_augment(self.load_image(image_gen, bg_gray)[0], bg_gray))
            images_mr.append(self._maybe_augment(self.load_image(image_gen.replace("_albedo", "_mr"), bg_gray)[0], bg_gray))
            images_normal.append(
                self._maybe_augment(self.load_image(image_gen.replace("_albedo", "_normal"), bg_gray)[0], bg_gray)
            )
            images_position.append(
                self._maybe_augment(self.load_image(image_gen.replace("_albedo", "_pos"), bg_gray)[0], bg_gray)
            )

        return {
            "images_cond": torch.stack(images_ref, dim=0).float(),
            "images_albedo": torch.stack(images_albedo, dim=0).float(),
            "images_mr": torch.stack(images_mr, dim=0).float(),
            "images_normal": torch.stack(images_normal, dim=0).float(),
            "images_position": torch.stack(images_position, dim=0).float(),
            "name": sample_dir,
        }
