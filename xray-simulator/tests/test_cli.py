# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI behavior with real preprocessing/display/effects and a CPU transport fixture."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from xray_simulator.cli import main
from xray_simulator.simulator import xray_simulator


@pytest.fixture
def cpu_transport(monkeypatch):
    calls = []

    def initialize(simulator):
        # Transport is the only replaced component. Configuration, preprocessing,
        # seeded intensity effects, display mapping and export use production code.
        class Renderer:
            def render(self, rotation, translation):
                calls.append((tuple(rotation), tuple(translation)))
                a = simulator.volume.mu_volume.sum(axis=0) * simulator.volume.spacing_zyx_mm[0]
                return (simulator.config.physics.i0 * np.exp(-a)).astype(np.float32)

        simulator._renderer = Renderer()

    monkeypatch.setattr(xray_simulator, "_init_renderer", initialize)
    return calls


def render(tmp_path, name, *options):
    output = tmp_path / name
    assert main(["render", "--synthetic", "--output", str(output), *options]) == 0
    return np.load(output / "frame_0000.npy"), json.loads((output / "run.json").read_text())


def test_dryrun_never_loads_data_or_initializes_renderer(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dryrun must not load voxels or touch the GPU")

    monkeypatch.setattr("xray_simulator.cli._load_volume", forbidden)
    monkeypatch.setattr(xray_simulator, "_init_renderer", forbidden)
    output = tmp_path / "preview"
    assert (
        main(
            [
                "render",
                "--synthetic",
                "--appearance",
                "xray",
                "--gamma",
                "1.7",
                "--gain",
                "1.2",
                "--poisson-photons",
                "1000",
                "--output",
                str(output),
                "--dryrun",
            ]
        )
        == 0
    )
    config = json.loads(capsys.readouterr().out)["simulator"]
    assert config["display"]["polarity"] == "diagnostic"
    assert config["display"]["gamma"] == 1.7
    assert config["realism"]["enabled"] is True
    assert not output.exists()


def test_piecewise_transfer_is_applied_and_saved(tmp_path):
    output = tmp_path / "cache"
    args = ["preprocess", "--synthetic", "--output", str(output), "--no-hu-clip"]
    for hu, mu in [(-1000, 0), (0, 0.01), (1000, 0.05)]:
        args += ["--hu-control-point", str(hu), str(mu)]
    assert main(args) == 0
    mu = np.load(output / "mu_volume.npy")
    np.testing.assert_allclose(np.unique(mu), [0.0, 0.0116, 0.046], atol=1e-7)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["hu_to_mu"]["control_points"] == [[-1000.0, 0.0], [0.0, 0.01], [1000.0, 0.05]]
    assert metadata["anatomical_frame"] == "LPS"


def test_hu_ramp_and_clipping_change_preprocessed_values(tmp_path):
    output = tmp_path / "clipped"
    main(
        [
            "preprocess",
            "--synthetic",
            "--hu-window-center",
            "100",
            "--hu-window-width",
            "200",
            "--mu-min",
            "0.001",
            "--mu-max",
            "0.021",
            "--hu-clip",
            "0",
            "100",
            "--output",
            str(output),
        ]
    )
    np.testing.assert_allclose(np.unique(np.load(output / "mu_volume.npy")), [0.001, 0.005, 0.011], atol=1e-7)


def test_polarity_gamma_and_explicit_preset_override(tmp_path, cpu_transport):
    fluoro, _ = render(tmp_path, "fluoro")
    xray, _ = render(tmp_path, "xray", "--appearance", "xray")
    np.testing.assert_allclose(xray, 1 - fluoro, atol=1e-7)
    gamma, _ = render(tmp_path, "gamma", "--gamma", "2")
    np.testing.assert_allclose(gamma, np.sqrt(fluoro), atol=1e-7)
    overridden, config = render(tmp_path, "override", "--appearance", "xray", "--polarity", "fluoro")
    np.testing.assert_array_equal(overridden, fluoro)
    assert config["simulator"]["display"]["polarity"] == "fluoro"


def test_scaling_windows_and_i0(tmp_path, cpu_transport):
    transmission, _ = render(tmp_path, "transmission", "--scaling", "transmission")
    doubled_i0, _ = render(tmp_path, "i0", "--scaling", "transmission", "--i0", "2")
    np.testing.assert_array_equal(transmission, doubled_i0)
    window, config = render(tmp_path, "window", "--window", ".4", ".9")
    np.testing.assert_allclose(window, np.clip((transmission - 0.4) / 0.5, 0, 1), atol=1e-7)
    assert config["simulator"]["display"]["scaling"] == "window"
    normalized, _ = render(tmp_path, "norm", "--scaling", "per_frame")
    assert normalized.min() == pytest.approx(0)
    assert normalized.max() == pytest.approx(1)
    default_log, _ = render(tmp_path, "log")
    narrow, _ = render(tmp_path, "narrow", "--log-window", "0", "2")
    assert np.mean(narrow) < np.mean(default_log)


def test_hu_settings_reach_rendering(tmp_path, cpu_transport):
    baseline, _ = render(tmp_path, "base")
    stronger, _ = render(tmp_path, "stronger", "--mu-max", ".04")
    assert np.mean(stronger) < np.mean(baseline)


def test_gain_bias_and_raw_intensity_output(tmp_path, cpu_transport):
    display, plan = render(
        tmp_path,
        "effects",
        "--gain",
        "0",
        "--bias",
        ".25",
        "--i0",
        "2",
        "--scaling",
        "transmission",
        "--keep-intensity",
    )
    np.testing.assert_allclose(display, 0.125)
    np.testing.assert_allclose(np.load(tmp_path / "effects/intensity_0000.npy"), 0.25)
    assert plan["simulator"]["realism"]["enabled"] is True


def test_seeded_cine_repeats_run_but_not_each_frame(tmp_path, cpu_transport):
    options = [
        "--frames",
        "3",
        "--seed",
        "17",
        "--poisson-photons",
        "500",
        "--gaussian-sigma",
        ".01",
        "--keep-intensity",
    ]
    render(tmp_path, "first", *options)
    render(tmp_path, "second", *options)
    first = [np.load(tmp_path / "first" / f"intensity_{i:04d}.npy") for i in range(3)]
    for i in range(3):
        np.testing.assert_array_equal(first[i], np.load(tmp_path / "second" / f"intensity_{i:04d}.npy"))
    assert not np.array_equal(first[0], first[1])


def test_blur_reduces_edge_energy(tmp_path, cpu_transport):
    sharp, _ = render(tmp_path, "sharp", "--scaling", "transmission")
    blurred, _ = render(tmp_path, "blur", "--scaling", "transmission", "--blur-sigma-px", "2")
    def energy(x):
        return np.sum(np.diff(x, axis=0) ** 2) + np.sum(np.diff(x, axis=1) ** 2)

    assert energy(blurred) < energy(sharp)


def test_calibration_is_frozen_and_recorded(tmp_path, cpu_transport):
    _, plan = render(tmp_path, "calibrated", "--calibrate-display", "1", "99", "--frames", "3")
    assert len(cpu_transport) == 4  # One calibration render, followed by three frames.
    assert plan["simulator"]["display"]["log_window"] != [0.0, 6.0]
    a = np.load(tmp_path / "calibrated/frame_0000.npy")
    np.testing.assert_array_equal(a, np.load(tmp_path / "calibrated/frame_0002.npy"))


def test_cache_render_and_hu_rejection(tmp_path, cpu_transport):
    cache = tmp_path / "cache"
    main(["preprocess", "--synthetic", "--output", str(cache)])
    main(["render", "--cache", str(cache), "--view", "ap", "--output", str(tmp_path / "valid")])
    with pytest.raises(SystemExit) as exc:
        main(["render", "--cache", str(cache), "--mu-max", ".05", "--output", str(tmp_path / "bad")])
    assert exc.value.code == 2
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--gamma", "nan"],
        ["--gamma", "0"],
        ["--i0", "inf"],
        ["--gaussian-sigma", "-1"],
        ["--detector-size", "0", "128"],
        ["--sdd-mm", "400", "--sid-mm", "500"],
        ["--window", ".8", ".2"],
        ["--window", "0", "2"],
        ["--hu-clip", "50", "20"],
        ["--log-window", "1", "1"],
        ["--scaling", "transmission", "--window", "0", "1"],
        ["--calibrate-display", "90", "10"],
        ["--seed", "-1"],
        ["--hu-control-point", "0", "0"],
        ["--hu-control-point", "0", "0", "--hu-control-point", "-1", ".01"],
        ["--hu-control-point", "0", "0", "--hu-control-point", "1000", ".01", "--mu-max", ".02"],
        ["--scatter", ".2"],
        ["--dose-mgy", "1"],
        ["--persistence", ".5"],
    ],
)
def test_invalid_controls_fail_before_rendering(tmp_path, monkeypatch, options):
    monkeypatch.setattr(xray_simulator, "_init_renderer", lambda *a: pytest.fail("GPU must not be initialized"))
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        main(["render", "--synthetic", "--output", str(output), *options])
    assert exc.value.code == 2
    assert not output.exists()


def test_existing_outputs_are_preserved(tmp_path):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(SystemExit):
        main(["preprocess", "--synthetic", "--output", str(tmp_path)])
    assert sentinel.read_text() == "keep"


def test_module_help_and_i4h_modes():
    result = subprocess.run(
        [sys.executable, "-m", "xray_simulator", "render", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--polarity" in result.stdout and "--hu-control-point" in result.stdout
    metadata = json.loads((Path(__file__).parents[1] / "metadata.json").read_text())
    for command in ("render", "preprocess"):
        assert metadata["application"]["modes"][command]["run"]["command"] == f"python -m xray_simulator {command}"
