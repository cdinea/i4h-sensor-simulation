# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line preprocessing and rendering with explicit simulator settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import (
    DISPLAY_PRESETS,
    CarmGeometry,
    DisplaySettings,
    HuToMuMapping,
    PreprocessingSettings,
    RealismSettings,
    SimulatorConfig,
    XrayPhysics,
)
from .geometry import VIEWS, view_frame_warning, view_rotation
from .preprocessor import VolumePreprocessor
from .simulator import Pose, xray_simulator
from .volume import PreprocessedVolume, VolumeMetadata


def finite(value: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def positive(value: str) -> float:
    result = finite(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value: str) -> float:
    result = finite(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def seed_value(value: str) -> int | None:
    if value == "random":
        return None
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative or 'random'")
    return result


def _add_input(parser, *, allow_cache):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Use a built-in HU phantom (LPS axes)")
    source.add_argument("--nifti", type=Path, help="CT NIfTI with values in HU; retains native array axes")
    source.add_argument("--dicom", type=Path, help="CT DICOM series directory; retains native array axes")
    if allow_cache:
        source.add_argument("--cache", type=Path, help="Existing preprocessed mu-volume directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument(
        "--dryrun",
        "--dry-run",
        action="store_true",
        help="Print resolved settings without loading voxels, using a GPU or writing files",
    )
    mapping = parser.add_argument_group("HU-to-attenuation mapping (raw HU sources only)")
    mapping.add_argument("--hu-window-center", type=finite, help="Transfer-function ramp center in HU")
    mapping.add_argument("--hu-window-width", type=positive, help="Transfer-function ramp width in HU")
    mapping.add_argument("--mu-min", type=nonnegative, help="Attenuation at the low HU endpoint, mm^-1")
    mapping.add_argument("--mu-max", type=nonnegative, help="Attenuation at the high HU endpoint, mm^-1")
    mapping.add_argument(
        "--hu-control-point",
        nargs=2,
        type=finite,
        action="append",
        metavar=("HU", "MU"),
        help="Repeat for a piecewise-linear curve; increasing HU and nonnegative mu",
    )
    clipping = mapping.add_mutually_exclusive_group()
    clipping.add_argument(
        "--hu-clip",
        nargs=2,
        type=finite,
        metavar=("MIN", "MAX"),
        help="HU clipping bounds before the transfer function",
    )
    clipping.add_argument("--no-hu-clip", action="store_true", help="Disable preprocessing HU clipping")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xray-simulator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preprocess = sub.add_parser("preprocess", help="Convert HU to a cached attenuation volume (CPU)")
    _add_input(preprocess, allow_cache=False)
    render = sub.add_parser("render", help="Render configured X-ray/fluoroscopy frames (CUDA/Slang)")
    _add_input(render, allow_cache=True)
    display = render.add_argument_group("Display")
    display.add_argument(
        "--appearance",
        choices=sorted(DISPLAY_PRESETS),
        default="fluoro",
        help="Base preset; explicit display options override it",
    )
    display.add_argument(
        "--polarity",
        choices=("fluoro", "diagnostic", "xray"),
        help="Dense dark (fluoro) or dense bright (diagnostic/xray)",
    )
    display.add_argument(
        "--scaling",
        choices=("log", "transmission", "window", "per_frame"),
        help="Intensity normalization; per_frame refits every frame",
    )
    windows = display.add_mutually_exclusive_group()
    windows.add_argument(
        "--log-window",
        nargs=2,
        type=nonnegative,
        metavar=("LOW", "HIGH"),
        help="Line-integral display interval; selects log scaling",
    )
    windows.add_argument(
        "--window",
        nargs=2,
        type=nonnegative,
        metavar=("LOW", "HIGH"),
        help="Transmission display interval in [0,1]; selects window scaling",
    )
    windows.add_argument(
        "--calibrate-display",
        nargs=2,
        type=nonnegative,
        metavar=("P_LOW", "P_HIGH"),
        help="Fit one log window to these first-view percentiles, then freeze it",
    )
    display.add_argument("--gamma", type=positive, help="Display gamma; >1 brightens midtones")
    sensor = render.add_argument_group("Intensity effects (uncalibrated sensor approximations)")
    sensor.add_argument(
        "--i0", type=positive, default=1.0, help="Incident intensity in arbitrary units; not physical dose or mAs"
    )
    sensor.add_argument("--gain", type=nonnegative, default=1.0, help="Intensity gain before noise; default 1")
    sensor.add_argument("--bias", type=finite, default=0.0, help="Additive intensity bias before noise; default 0")
    sensor.add_argument(
        "--poisson-photons",
        type=nonnegative,
        default=0.0,
        help="Photon count scale at intensity=1; 0 disables quantum-noise approximation",
    )
    sensor.add_argument(
        "--gaussian-sigma", type=nonnegative, default=0.0, help="Additive Gaussian noise sigma in intensity units"
    )
    sensor.add_argument("--blur-sigma-px", type=nonnegative, default=0.0, help="Gaussian detector blur sigma in pixels")
    sensor.add_argument(
        "--seed",
        type=seed_value,
        default=0,
        help="Nonnegative random seed, or random; cine increments the seed per frame",
    )
    geometry = render.add_argument_group("Camera and integration")
    geometry.add_argument(
        "--detector-size", nargs=2, type=positive_int, default=(512, 512), metavar=("WIDTH", "HEIGHT")
    )
    geometry.add_argument("--pixel-spacing-mm", type=positive, default=0.5)
    geometry.add_argument("--sdd-mm", type=positive, default=1020.0, help="Source-to-detector distance")
    geometry.add_argument("--sid-mm", type=positive, default=510.0, help="Source-to-isocenter distance")
    geometry.add_argument("--step-mm", type=positive, default=0.5, help="Ray-march integration step")
    pose = geometry.add_mutually_exclusive_group()
    pose.add_argument("--view", choices=VIEWS, help="Named clinical view; requires LPS volume metadata")
    pose.add_argument(
        "--rotation-deg",
        nargs=3,
        type=finite,
        metavar=("RX", "RY", "RZ"),
        help="Raw Euler angles, ZXY convention; default 0 0 0 in native volume axes",
    )
    geometry.add_argument(
        "--translation-mm",
        nargs=3,
        type=finite,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Isocenter displacement from the volume center",
    )
    output = render.add_argument_group("Output")
    output.add_argument("--frames", type=positive_int, default=1, help="Number of frames at the specified pose")
    output.add_argument(
        "--fps", type=positive, default=15.0, help="Sequence metadata rate; no timing, motion or dose model"
    )
    output.add_argument(
        "--format", choices=("npy", "png", "both"), default="npy", help="Float32 NPY, 8-bit grayscale PNG, or both"
    )
    output.add_argument(
        "--keep-intensity", action="store_true", help="Also save pre-display intensity arrays after enabled effects"
    )
    return parser


_HU_OPTIONS = ("hu_window_center", "hu_window_width", "mu_min", "mu_max", "hu_control_point", "hu_clip")


def preprocessing_settings(args) -> PreprocessingSettings | None:
    explicit = any(getattr(args, key) is not None for key in _HU_OPTIONS) or args.no_hu_clip
    if getattr(args, "cache", None):
        if explicit:
            raise ValueError("HU options cannot modify a cached mu-volume; preprocess the original HU source again")
        return None
    defaults = HuToMuMapping()
    if args.hu_control_point is not None:
        if any(getattr(args, key) is not None for key in _HU_OPTIONS[:4]):
            raise ValueError("--hu-control-point cannot be combined with ramp window/endpoints")
        mapping = HuToMuMapping(control_points=tuple(tuple(p) for p in args.hu_control_point))
    else:
        mapping = HuToMuMapping.from_window_level(
            defaults.window_center if args.hu_window_center is None else args.hu_window_center,
            defaults.window_width if args.hu_window_width is None else args.hu_window_width,
            mu_min=defaults.mu_min if args.mu_min is None else args.mu_min,
            mu_max=defaults.mu_max if args.mu_max is None else args.mu_max,
        )
    settings = PreprocessingSettings(hu_to_mu=mapping, clip_hu=not args.no_hu_clip)
    if args.hu_clip is not None:
        low, high = args.hu_clip
        if low >= high:
            raise ValueError("--hu-clip requires MIN < MAX")
        settings = replace(settings, hu_clip_min=low, hu_clip_max=high)
    return settings


def simulator_settings(args) -> tuple[SimulatorConfig, Pose]:
    display = DisplaySettings.preset(args.appearance)
    updates = {key: getattr(args, key) for key in ("polarity", "scaling", "gamma") if getattr(args, key) is not None}
    if updates.get("polarity") == "xray":
        updates["polarity"] = "diagnostic"
    for option, mode in (("log_window", "log"), ("window", "window"), ("calibrate_display", "log")):
        value = getattr(args, option)
        if value is not None:
            if args.scaling is not None and args.scaling != mode:
                raise ValueError(f"--{option.replace('_', '-')} requires --scaling {mode}")
            updates["scaling"] = mode
            if option != "calibrate_display":
                updates[option] = tuple(value)
    if args.calibrate_display is not None:
        low, high = args.calibrate_display
        if not 0 <= low < high <= 100:
            raise ValueError("Calibration percentiles must satisfy 0 <= LOW < HIGH <= 100")
    display = replace(display, **updates)
    if args.sid_mm >= args.sdd_mm:
        raise ValueError("Geometry requires SID < SDD")
    realism = RealismSettings(
        enabled=(
            args.gain != 1
            or args.bias != 0
            or args.poisson_photons > 0
            or args.gaussian_sigma > 0
            or args.blur_sigma_px > 0
        ),
        gain=args.gain,
        bias=args.bias,
        poisson_photons=args.poisson_photons,
        gaussian_sigma=args.gaussian_sigma,
        blur_sigma_px=args.blur_sigma_px,
        seed=args.seed,
    )
    config = SimulatorConfig(
        geometry=CarmGeometry(
            source_to_detector_mm=args.sdd_mm,
            source_to_isocenter_mm=args.sid_mm,
            detector_width_px=args.detector_size[0],
            detector_height_px=args.detector_size[1],
            pixel_spacing_mm=args.pixel_spacing_mm,
        ),
        physics=XrayPhysics(step_mm=args.step_mm, i0=args.i0),
        display=display,
        realism=realism,
    ).with_output(keep_intensity=args.keep_intensity)
    translation = tuple(args.translation_mm)
    if args.view:
        pose = Pose(rotation=view_rotation(args.view), translation=translation, view=args.view)
    else:
        pose = Pose(rotation=tuple(np.radians(args.rotation_deg or (0.0, 0.0, 0.0))), translation=translation)
    return config, pose


def _source(args) -> dict:
    for kind in ("cache", "nifti", "dicom"):
        path = getattr(args, kind, None)
        if path is not None:
            path = path.expanduser().resolve()
            if (kind in ("cache", "dicom") and not path.is_dir()) or (kind == "nifti" and not path.is_file()):
                raise ValueError(f"{kind} input does not exist: {path}")
            if kind == "cache" and not all((path / name).is_file() for name in ("metadata.json", "mu_volume.npy")):
                raise ValueError("Cache must contain metadata.json and mu_volume.npy")
            return {"kind": kind, "path": str(path)}
    return {"kind": "synthetic", "path": None}


def _load_volume(source, settings):
    kind, path = source["kind"], source["path"]
    if kind == "cache":
        volume = PreprocessedVolume.load(path)
    else:
        if kind == "synthetic":
            z, y, x = np.mgrid[:64, :64, :64] - 31.5
            radius = np.sqrt(x * x + y * y + z * z)
            hu = np.full(radius.shape, -1000.0, dtype=np.float32)
            hu[radius < 25] = 40.0
            hu[(x - 8) ** 2 + (y + 5) ** 2 + z * z < 8**2] = 900.0
            preprocessor = VolumePreprocessor.from_numpy(
                hu, settings=settings, spacing_zyx_mm=(1.0, 1.0, 1.0), anatomical_frame="LPS"
            )
        elif kind == "nifti":
            preprocessor = VolumePreprocessor.from_nifti(path, settings=settings)
        else:
            preprocessor = VolumePreprocessor.from_dicom(path, settings=settings)
        volume = preprocessor.preprocess()
    if not np.isfinite(volume.mu_volume).all() or np.any(volume.mu_volume < 0):
        raise ValueError("Attenuation volume must be finite and nonnegative")
    return volume


def _save_frames(cine, output, image_format):
    if image_format in ("png", "both"):
        from PIL import Image
    for frame in cine:
        stem = output / f"frame_{frame.frame_idx:04d}"
        if image_format in ("npy", "both"):
            np.save(stem.with_suffix(".npy"), frame.image)
        if image_format in ("png", "both"):
            Image.fromarray(np.rint(np.clip(frame.image, 0, 1) * 255).astype(np.uint8)).save(stem.with_suffix(".png"))
        if frame.intensity is not None:
            np.save(output / f"intensity_{frame.frame_idx:04d}.npy", frame.intensity)


def run(args) -> dict:
    settings = preprocessing_settings(args)
    source = _source(args)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output directory: {output}")
    plan = {
        "command": args.command,
        "source": source,
        "output": str(output),
        "preprocessing": asdict(settings) if settings is not None else None,
    }
    cached_metadata = None
    if source["kind"] == "cache":
        cached_metadata = VolumeMetadata.from_dict(json.loads((Path(source["path"]) / "metadata.json").read_text()))
        plan["volume_metadata"] = cached_metadata.to_dict()
    if args.command == "render":
        config, pose = simulator_settings(args)
        known_frame = (
            "LPS" if source["kind"] == "synthetic" else (cached_metadata.anatomical_frame if cached_metadata else None)
        )
        warning = view_frame_warning(pose.view, known_frame)
        if warning:
            raise ValueError(warning + " Use --rotation-deg in the source's native axes, or supply an LPS cache.")
        plan.update(
            simulator=asdict(config),
            pose=pose.to_dict(),
            frames=args.frames,
            fps=args.fps,
            format=args.format,
            calibration_percentiles=args.calibrate_display,
        )
    if args.dryrun:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    if args.command == "render" and args.format in ("png", "both"):
        try:
            import PIL.Image  # noqa: F401
        except ImportError as exc:
            raise ValueError("PNG output requires Pillow; install xray-simulator[cli]") from exc
    volume = _load_volume(source, settings)
    plan["volume_metadata"] = volume.metadata.to_dict()
    if args.command == "preprocess":
        output.mkdir(parents=True, exist_ok=False)
        volume.save(output)
    else:
        simulator = xray_simulator(volume, config)
        if args.calibrate_display is not None:
            simulator.calibrate_display(pose=pose, percentiles=tuple(args.calibrate_display))
        plan["simulator"]["display"] = asdict(simulator.display)
        # render_cine advances each frame's seed; repeated render_frame calls would
        # reuse the same noise realization with a fixed seed.
        cine = simulator.render_cine([pose] * args.frames, fps=args.fps, progress=False)
        output.mkdir(parents=True, exist_ok=False)
        _save_frames(cine, output, args.format)
    (output / "run.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.command} output to {output}")
    return plan


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0
