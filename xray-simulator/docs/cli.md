# Configuring X-ray and fluoroscopy from the CLI

The `xray-simulator` command and `python -m xray_simulator` provide CPU
`preprocess` and CUDA/Slang `render` commands. The `./i4h` wrapper exposes them as
`preprocess` and `render` modes. Existing example modes remain available.

## Quick start

From the repository root, install the package and the optional PNG/rendering dependencies:

```bash
python -m pip install -e './xray-simulator[cli,slang]'
python -m xray_simulator render --help
```

Preview the resolved settings, then render a synthetic phantom:

```bash
python -m xray_simulator render --synthetic --view ap \
  --appearance fluoro --log-window 0 3 --gamma 1.2 \
  --output /tmp/fluoro-example --format both --keep-intensity --dryrun

python -m xray_simulator render --synthetic --view ap \
  --appearance fluoro --log-window 0 3 --gamma 1.2 \
  --output /tmp/fluoro-example --format both --keep-intensity
```

Use a new output directory for each run. The CLI rejects an existing output path.
`--dryrun` validates argument combinations and input paths and prints the plan;
it does not load voxel arrays, initialize a GPU, or write files. It cannot prove
that a source volume can be decoded or that CUDA is available.

Through the container wrapper, use `--run-args`:

```bash
./i4h run xray-simulator render --dryrun --verbose \
  --run-args="--synthetic --view ap --appearance xray --output output/xray-example --format both"

./i4h run xray-simulator render \
  --run-args="--synthetic --view ap --appearance xray --output output/xray-example --format both"
```

The wrapper's outer `--dryrun` previews Docker commands. An inner
`--run-args="... --dryrun"` previews the simulator settings when the application
is actually launched. Use container-visible input paths; output under
`output/` is retained in the mounted `xray-simulator/output/` directory.

## RQ2 parameter coverage

| Requirement | CLI controls | Status |
| --- | --- | --- |
| Display polarity | `--appearance fluoro`, `--appearance xray`, `--polarity fluoro/diagnostic/xray` | Implemented; display convention only |
| Intensity scaling and normalization | `--scaling log/transmission/window/per_frame` | Implemented |
| HU-to-attenuation transfer function | `--hu-window-center`, `--hu-window-width`, `--mu-min`, `--mu-max`, repeated `--hu-control-point HU MU` | Implemented |
| HU clipping | `--hu-clip MIN MAX`, `--no-hu-clip` | Implemented |
| Gain and offset | `--gain`, `--bias` | Implemented in intensity space |
| Quantum and electronic-noise approximations | `--poisson-photons`, `--gaussian-sigma`, `--seed` | Implemented; uncalibrated noise models |
| Incident intensity / exposure surrogate | `--i0` | Implemented in arbitrary units; not dose, mAs or kVp |
| Windowing, contrast and gamma | `--log-window`, `--window`, `--calibrate-display`, `--gamma` | Implemented |
| Detector blur | `--blur-sigma-px` | Implemented Gaussian approximation |
| Geometry and sampling | `--detector-size`, `--pixel-spacing-mm`, `--sdd-mm`, `--sid-mm`, `--step-mm` | Implemented |
| Sensor dynamic range / ADC bit depth | Float NPY or 8-bit PNG export | Storage choices only; no physical detector saturation/ADC model |
| Sharpening | — | P1 follow-up; no model exposed |
| Scatter | — | P1 follow-up; no scatter model |
| Temporal filtering and persistence | — | P1 follow-up; independent frames today |
| Calibrated dose settings / AEC | — | P1 follow-up; no dose or exposure-control model |

Unsupported effects are not accepted as placeholder flags. The CLI exposes
existing simulated effects; it does not establish their physical calibration.

## Display behavior

Explicit display options override `--appearance`. `--polarity` changes polarity
without replacing the preset's other settings. At gamma 1, fluoroscopy and
X-ray displays of the same intensity are complements.

- `log`: map `-ln(I/i0)` through `--log-window LOW HIGH` (default 0–6).
- `transmission`: display `I/i0`, clipped to [0,1].
- `window`: stretch a transmission interval, `--window LOW HIGH`, within [0,1].
- `per_frame`: normalize each frame by its own extrema; brightness can change
  with the field of view and noise.

`--window` selects window scaling and `--log-window` selects log scaling.
An explicit conflicting `--scaling` is rejected. Narrower windows increase
contrast and clipping. `--gamma 1` is linear; values above 1 brighten midtones.

`--calibrate-display 1 99` measures one noise-free render at the requested pose,
sets the log window from its attenuation percentiles, and freezes that window
for the output sequence. The fitted values are saved in `run.json`. This is
patient-specific display fitting, not detector calibration or a physical AEC model.

## HU transfer functions and caches

Preprocess a synthetic phantom with a piecewise-linear transfer function:

```bash
python -m xray_simulator preprocess --synthetic \
  --hu-control-point -1000 0 --hu-control-point 0 0.01 \
  --hu-control-point 1000 0.05 --output /tmp/custom-mu --dryrun

python -m xray_simulator preprocess --synthetic \
  --hu-control-point -1000 0 --hu-control-point 0 0.01 \
  --hu-control-point 1000 0.05 --output /tmp/custom-mu

python -m xray_simulator render --cache /tmp/custom-mu --view ap \
  --appearance fluoro --output /tmp/custom-mu-frames --dryrun

python -m xray_simulator render --cache /tmp/custom-mu --view ap \
  --appearance fluoro --output /tmp/custom-mu-frames
```

Control points require at least two knots, increasing HU and nonnegative
attenuation in mm⁻¹. Values between knots are interpolated; values outside
are clamped to the endpoint attenuation. Ramp controls instead define the HU
window and attenuation endpoints. Combining these two definitions is rejected.

HU clipping happens before the transfer function. Defaults match the Python API:
HU clip [-1024,3071], transfer ramp HU [-1000,3000] to attenuation [0,0.02] mm⁻¹.
The chosen mapping is saved in the cache metadata and run record.

`render` accepts `--synthetic`, `--nifti`, `--dicom`, or `--cache`. HU options work
on raw sources for either command. They are rejected with `--cache`, because a
mu-volume no longer contains the original HU values. Reprocess the source to
change its transfer function.

NIfTI input requires nibabel; DICOM input requires SimpleITK (both are included
in the package's `all` extra). These commands use the existing native-axis
loaders. Named `--view` values require a cache labeled LPS; the synthetic phantom
is labeled LPS. For other volumes use explicitly chosen `--rotation-deg RX RY RZ`
in the source's native array frame or prepare an LPS cache externally. The default
raw rotation is zero, not a named AP projection. Rotations use the renderer's
ZXY convention. `--translation-mm X Y Z` displaces the isocenter from volume center.

## Sensor effects and repeated frames

```bash
python -m xray_simulator render --synthetic --view ap \
  --gain 1.1 --bias 0.005 --poisson-photons 2000 \
  --gaussian-sigma 0.003 --blur-sigma-px 0.7 --seed 42 \
  --frames 10 --fps 15 --keep-intensity --output /tmp/fluoro-noise --dryrun

python -m xray_simulator render --synthetic --view ap \
  --gain 1.1 --bias 0.005 --poisson-photons 2000 \
  --gaussian-sigma 0.003 --blur-sigma-px 0.7 --seed 42 \
  --frames 10 --fps 15 --keep-intensity --output /tmp/fluoro-noise
```

Nondefault gain/bias/noise/blur options automatically enable intensity effects.
The order is gain/bias, Poisson noise, Gaussian noise, Gaussian blur, then display
scaling/polarity/gamma. Poisson counts have expectation
`max(gain*I + bias, 0) * poisson_photons`; the result is divided by the photon
scale to return to intensity units. Consequently both `--i0` and the photon scale
can change noise statistics. Increasing `i0` alone does not change the noiseless
normalized display because it divides out of `I/i0`.

`--frames` repeats the specified pose. A fixed seed reproduces the run while
advancing the seed per frame; `--seed random` requests unseeded draws. These are
independent noise realizations, not temporal detector noise or persistence.
`--fps` records sequence metadata only; it does not change photon counts or dose.

## Output and reproducibility

Each render directory contains numbered display frames and `run.json` with the
resolved preprocessing, display, physics, effects, geometry, pose, seed and
sequence settings. Cache runs include the original volume metadata and HU mapping.
`--keep-intensity` adds the pre-display intensity after enabled effects, in
floating-point NPY files. PNG output requires the `cli` extra and is 8-bit
grayscale; NPY preserves float32 values. Neither format choice changes the
simulated detector physics.
