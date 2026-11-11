"""
Run EMRI population detectability experiments with broad and/or narrow priors.

For each realization, this script:
  1. Draws a formation channel with probability 1/2 for each channel.
  2. Draws the injected final eccentricity from that channel's population.
  3. Draws one noise seed and generates eccentric, circular, and noise-only data using the same noise SFT realization.
  4. Fits eccentric and circular templates to the data with the requested priors.
  5. Saves injection metadata once per T_SFT, and saves all fit results in HDF5.

The broad results retain the original population script's layout:
  results/run_XXXXXX/tsft_XXXXXX/{eccentric_signal,circular_signal,noise_only}/

Narrow-prior results are nested under:
  results/run_XXXXXX/tsft_XXXXXX/narrow_bounds/{eccentric_signal,circular_signal}/

Likelihood dataset names are kept consistent in both sections:
  logL_ecc, logL_circ, lambda_LR

Example
-------

python run_det_stats_pop.py \
    --n_runs 1 \
    --rng_seed 1234 \
    --mode_selection "fixed modes" \
    --specific_modes 2,0 \
    --vals_t_sft 172800 \
    --tdi_flag \
    --outfile pop_stats.h5

"""

from __future__ import annotations
import argparse
import json
import logging
import multiprocessing as mp
import secrets
import time
from functools import partial
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import pandas as pd

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.ylm import GetYlms

from utils import PSD
from gw_signal import generate_injection_data
from inspiral import find_relevant_modes
from search import _negative_log_likelihood, _run_differential_evolution
from file_handling import load_population

PARAM_NAMES = ["log10_m1", "m2", "a", "Tpl", "ef"]
SOURCE_PARAM_NAMES = ["m1", "m2", "a", "Tpl", "ef", "xI"]
CHANNEL_RELAXATION = "Two-Body Relaxation"
CHANNEL_HILLS = "Hills Mechanism"

traj_gen = EMRIInspiral(func=KerrEccEqFlux)
amp_gen = AmpInterpKerrEccEq()
ylm_gen = GetYlms()

logger = logging.getLogger("run_det_stats_pop")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run EMRI population experiments with broad and/or narrow priors.")
    
    # Population inputs
    parser.add_argument("--relaxation_file", type=str, default="relaxation_channel.csv", help="CSV for the two-body-relaxation population (requires epl and weight).")
    parser.add_argument("--hills_file", type=str, default="hills_mechanism.csv", help="CSV for the Hills-mechanism population (requires epl).")
    
    # Reference/injection parameters.
    parser.add_argument("--m1", type=float, default=4e6, help="Primary mass [solar masses].")
    parser.add_argument("--m2", type=float, default=10.0, help="Secondary mass [solar masses].")
    parser.add_argument("--a", type=float, default=0.0, help="Dimensionless spin parameter.")
    parser.add_argument("--Tpl", type=float, default=0.9, help="Time to plunge [years].")
    parser.add_argument("--xI", type=float, default=1.0, help="Cosine of inclination used for mode selection and metadata.")
    parser.add_argument("--theta_K", type=float, default=np.pi / 3, help="Spin-orientation polar angle [radians].")
    parser.add_argument("--phi_K", type=float, default=0.0, help="Spin-orientation azimuthal angle [radians].")

    # Bounds: broad bounds are absolute; narrow bounds are relative to the reference m1, m2, a, and Tpl. Both use the same global final-eccentricity range.
    parser.add_argument("--ef_bounds", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.0, 0.81), help="Prior range for final eccentricity (used by both prior choices).")
    parser.add_argument("--dev", type=float, default=1e-2, help="Relative half-width for narrow m1, m2, a, and Tpl bounds.")
    parser.add_argument("--m1_bounds", nargs=2, type=float, metavar=("MIN", "MAX"), default=(5e5, 5e6), help="Broad bounds for m1 [solar masses].")
    parser.add_argument("--m2_bounds", nargs=2, type=float, metavar=("MIN", "MAX"), default=(1.0, 100.0), help="Broad bounds for m2 [solar masses].")
    parser.add_argument("--a_bounds", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.0, 0.0), help="Broad bounds for dimensionless spin a.")
    parser.add_argument("--Tpl_bounds", nargs=2, type=float, metavar=("MIN", "MAX"), default=(0.1, 1.0), help="Broad bounds for Tpl [years].")

    # Mode selection.
    parser.add_argument("--mode_selection", choices=("fixed modes", "dominant mode", "threshold"), default="fixed modes", help="Strategy used to select waveform modes.")
    parser.add_argument("--specific_modes", nargs="+", default=["2,0"], help="Modes for fixed selection, e.g. --specific_modes 2,0 3,0.")
    parser.add_argument("--threshold", type=float, default=0.99, help="Cumulative-SNR threshold for threshold/dominant selection.")

    # Observation and SFT settings.
    parser.add_argument("--vals_t_sft", nargs="+", type=float, required=True, help="SFT durations [seconds] to run.")
    parser.add_argument("--t_data", type=float, default=1.0, help="Total observation duration [years].")
    parser.add_argument("--delta_t", type=float, default=5.0, help="Time step [seconds].")
    parser.add_argument("--snr_ref", type=float, default=50.0, help="Reference SNR for signal injection.")
    parser.add_argument("--t_snr", type=float, default=1.0, help="Reference duration for SNR scaling [years].")

    # Execution
    parser.add_argument("--n_runs", type=int, default=1, help="Number of population/noise realizations to run.")
    parser.add_argument("--workers", type=int, default=1, help="Number of multiprocessing workers for differential evolution; -1 uses all CPUs.")
    parser.add_argument("--tdi_flag", action="store_true", help="Use the TDI PSD.")
    parser.add_argument("--rng_seed", type=int, default=None, help="Optional master seed for reproducibility. If omitted, a random seed is generated and saved in the HDF5 output. Use the recorded seed to reproduce a run.")

    # Output
    parser.add_argument("--repo", type=str, default="../results", help="Directory in which to write the output HDF5 file.")
    parser.add_argument("--outfile", type=str, required=True, help="Output HDF5 filename (appends runs if the file already exists).")

    return parser.parse_args()

def draw_population_sample(rng, ef_rel, weight_rel, ef_hills):
    """Choose a channel 50/50 and draw e_f from the selected population."""

    if rng.random() < 0.5:
        idx = int(rng.choice(len(ef_rel), p=weight_rel))
        return CHANNEL_RELAXATION, float(ef_rel[idx])

    idx = int(rng.integers(0, len(ef_hills)))
    return CHANNEL_HILLS, float(ef_hills[idx])

def make_broad_bounds(args):
    """Create absolute broad prior bounds in optimizer-parameter order."""

    ef_min, ef_max = map(float, args.ef_bounds)

    return np.asarray([(np.log10(args.m1_bounds[0]), np.log10(args.m1_bounds[1])),
                       (args.m2_bounds[0], args.m2_bounds[1]),
                       (args.a_bounds[0], args.a_bounds[1]),
                       (args.Tpl_bounds[0], args.Tpl_bounds[1]),
                       (ef_min, ef_max)], dtype=float)

def make_narrow_bounds(args, m1, m2, a, tpl):
    """Create narrow bounds consistent with the narrow-prior backfill script."""

    dev = float(args.dev)
    ef_min, ef_max = map(float, args.ef_bounds)

    return np.asarray([(np.log10(m1 * (1.0 - dev)), np.log10(m1 * (1.0 + dev))),
                       (m2 * (1.0 - dev), m2 * (1.0 + dev)), 
                       (a * (1.0 - dev), a * (1.0 + dev)),
                       (tpl * (1.0 - dev), tpl * (1.0 + dev)),
                       (ef_min, ef_max)], dtype=float)

def parse_mode(mode_spec):

    cleaned = mode_spec.strip().strip("[]() ")
    pieces = [piece.strip() for piece in cleaned.split(",") if piece.strip()]

    return int(pieces[0]), int(pieces[1])

def choose_modes(args, ef_inj, psd):

    if args.mode_selection == "fixed modes":
        modes = [parse_mode(mode) for mode in args.specific_modes]

    else:
        all_modes, _, _, _ = find_relevant_modes(args.m1, args.m2, args.a, args.Tpl, ef_inj, args.xI, 1.0,
                                                args.theta_K, args.phi_K, psd, args.threshold,
                                                traj_gen, amp_gen, ylm_gen)
        modes = list(all_modes)

        if args.mode_selection == "dominant mode":
            modes = modes[:1]

    return modes

def make_objective(data_sfts, t_alpha, t_sft, psd, modes, fixed_ef=None):
    return partial(_negative_log_likelihood, t_alpha=t_alpha, data_sfts=data_sfts, T_SFT=t_sft, psd=psd, modes=modes, fixed_ef=fixed_ef)

def run_fit(data_sfts, t_alpha, full_bounds, model, t_sft, psd, modes, de_workers):
    """Fit one model; let the differential-evolution routine initialize itself."""

    if model == "eccentric":
        fit_bounds = np.asarray(full_bounds, dtype=float)
        objective = make_objective(data_sfts, t_alpha, t_sft, psd, modes)
    elif model == "circular":
        fit_bounds = np.asarray(full_bounds[:4], dtype=float)
        objective = make_objective(data_sfts, t_alpha, t_sft, psd, modes, fixed_ef=0.0)

    return _run_differential_evolution(objective, fit_bounds, de_workers)

def save_fit_result(data_group, model, result):
    """Save likelihood, full best-fit vector, and optimizer diagnostics."""

    logl_key = "logL_ecc" if model == "eccentric" else "logL_circ"
    fit_group_name = f"fit_{model}"
    best_params = np.asarray(result.x, dtype=float)

    if model == "circular":
        best_params = np.concatenate([best_params, [0.0]])

    fit_group = data_group.create_group(fit_group_name)
    ds = fit_group.create_dataset("best_params", data=best_params)
    ds.attrs["param_names"] = np.asarray(PARAM_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
    fit_group.attrs["optimizer_success"] = bool(getattr(result, "success", False))
    fit_group.attrs["optimizer_message"] = str(getattr(result, "message", ""))
    logl = -float(result.fun)
    data_group.create_dataset(logl_key, data=logl)
    data_group.file.flush()

    return logl

def save_parameter_value(group, key, value):
    """Save scalar or array-valued injection metadata without silently dropping it."""

    if value is None:
        group.attrs[f"{key}__is_none"] = True
        return

    try:
        array = np.asarray(value)

    except Exception:
        group.attrs[key] = repr(value)
        return

    if array.ndim == 0:
        scalar = array.item()

        if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8", errors="replace")

        if isinstance(scalar, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            group.attrs[key] = scalar

        else:
            group.attrs[key] = repr(scalar)

        return

    if array.dtype.kind in "biufc":
        group.create_dataset(key, data=array)

    elif array.dtype.kind in "SU":
        text_array = np.asarray(array, dtype=str)
        group.create_dataset(key, data=text_array.astype(h5py.string_dtype(encoding="utf-8")))

    else:
        group.attrs[key] = json.dumps(array.tolist(), default=str)

def save_generated_injection_parameters(parent, label, generated):
    """Save the generation parameter dictionary under the single injection metadata tree."""

    injection_group = parent.create_group(label)
    injection_dict = generated.get("injection_dict", {})
    param_dict = injection_dict.get("param_dict", {}) if isinstance(injection_dict, dict) else {}

    if isinstance(param_dict, dict):
        for key, value in param_dict.items():
            save_parameter_value(injection_group, str(key), value)

def save_injection_metadata(tsft_group, args, ef_inj, generated_ecc, generated_circ):
    """Save the injected source information once, separately from all model fits."""

    injection_group = tsft_group.create_group("injection_parameters")
    injection_group.attrs["t_data_years"] = float(args.t_data)
    injection_group.attrs["delta_t_seconds"] = float(args.delta_t)
    injection_group.attrs["snr_ref"] = float(args.snr_ref)
    injection_group.attrs["t_snr_years"] = float(args.t_snr)
    injection_group.attrs["tdi_flag"] = bool(args.tdi_flag)
    injection_group.attrs["mode_selection_theta_K"] = float(args.theta_K)
    injection_group.attrs["mode_selection_phi_K"] = float(args.phi_K)
    source_values = np.asarray([[args.m1, args.m2, args.a, args.Tpl, float(ef_inj), args.xI],
                                [args.m1, args.m2, args.a, args.Tpl, 0.0, args.xI],], dtype=float)
    source_ds = injection_group.create_dataset("base_source_parameters", data=source_values)
    source_ds.attrs["row_names"] = np.asarray(["eccentric_signal", "circular_signal"], dtype=h5py.string_dtype(encoding="utf-8"))
    source_ds.attrs["param_names"] = np.asarray(SOURCE_PARAM_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))
    save_generated_injection_parameters(injection_group, "eccentric_signal", generated_ecc)
    save_generated_injection_parameters(injection_group, "circular_signal", generated_circ)

def save_prior_bounds(tsft_group, bounds_by_name):

    prior_group = tsft_group.create_group("prior_bounds")

    for name, bounds in bounds_by_name.items():
        ds = prior_group.create_dataset(name, data=np.asarray(bounds, dtype=float))
        ds.attrs["param_names"] = np.asarray(PARAM_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))

def get_or_make_narrow_group(tsft_group, args):
    """Create the narrow result group and record the exact settings used."""

    group = tsft_group.require_group("narrow_bounds")
    config = {"dev": float(args.dev),
              "ef_bounds": [float(v) for v in args.ef_bounds],
              "t_data_years": float(args.t_data),
              "delta_t_seconds": float(args.delta_t),
              "snr_ref": float(args.snr_ref),
              "t_snr_years": float(args.t_snr),
              "tdi_flag": bool(args.tdi_flag)}
    group.attrs["prior_config_json"] = json.dumps(config, sort_keys=True)
    group.attrs["bounds_description"] = ("log10(m1) uses log10(m1_ref*(1 +/- dev)); m2, a and Tpl use ref*(1 +/- dev); ef uses the global ef_bounds. Optimizer initialization is delegated to differential evolution.")

    return group

def fit_dataset(data_group, data_sfts, t_alpha, bounds,prior_name, data_name, t_sft, psd, modes, de_workers):
    """Fit eccentric and circular models to one data realization and save both."""

    logl_values = {}

    for model_name in ("eccentric", "circular"):
        logger.info("%s: %s prior, %s data, %s fit", data_group.name, prior_name, data_name, model_name)
        result = run_fit(data_sfts, t_alpha, bounds, model_name, t_sft, psd, modes, de_workers)
        logl_values[model_name] = save_fit_result(data_group, model_name, result)

    data_group.create_dataset("lambda_LR", data=2.0 * (logl_values["eccentric"] - logl_values["circular"]))
    data_group.file.flush()

def run_one_tsft(run_group, t_sft, args, psd, modes, noise_seed, ef_inj, de_workers):
    """Generate the shared data once, then perform all requested fits."""

    tsft_name = f"tsft_{int(round(t_sft))}"
    tsft_group = run_group.create_group(tsft_name)
    tsft_group.create_dataset("t_sft", data=float(t_sft))
    ef_min, ef_max = map(float, args.ef_bounds)

    eccentric_injection = {"m1": args.m1, "m2": args.m2, "a": args.a, "Tpl": args.Tpl, "ef": float(ef_inj), "xI": float(args.xI)}
    circular_injection = dict(eccentric_injection)
    circular_injection["ef"] = 0.0

    generated_ecc = generate_injection_data(eccentric_injection, args.t_data, t_sft, args.delta_t, args.snr_ref, args.t_snr, psd, noise=True, seed=noise_seed)
    generated_circ = generate_injection_data(circular_injection, args.t_data, t_sft, args.delta_t, args.snr_ref, args.t_snr, psd, noise=True, seed=noise_seed)

    noise_sfts = np.asarray(generated_ecc["noise_sfts"], dtype=np.complex128)
    signal_ecc_sfts = np.asarray(generated_ecc["signal_sfts"], dtype=np.complex128)
    signal_circ_sfts = np.asarray(generated_circ["signal_sfts"], dtype=np.complex128)
    t_alpha = np.asarray(generated_ecc["t_alpha"])

    data_by_name = {"eccentric_signal": signal_ecc_sfts + noise_sfts,
                    "circular_signal": signal_circ_sfts + noise_sfts,
                    "noise_only": noise_sfts}

    save_injection_metadata(tsft_group, args, ef_inj, generated_ecc, generated_circ)
    bounds_by_name: dict[str, np.ndarray] = {}

    bounds_by_name["broad"] = make_broad_bounds(args)
    bounds_by_name["narrow"] = make_narrow_bounds(args, args.m1, args.m2, args.a, args.Tpl)

    save_prior_bounds(tsft_group, bounds_by_name)

    for prior_name, bounds in bounds_by_name.items():

        if prior_name == "broad":
            prior_parent = tsft_group

        else:
            prior_parent = get_or_make_narrow_group(tsft_group, args)
        
        dataset_names = (("eccentric_signal", "circular_signal", "noise_only") if prior_name == "broad" else ("eccentric_signal", "circular_signal"))

        for data_name in dataset_names:

            data_group = prior_parent.create_group(data_name)

            fit_dataset(data_group=data_group, data_sfts=data_by_name[data_name], t_alpha=t_alpha, bounds=bounds, 
                        prior_name=prior_name, data_name=data_name, t_sft=t_sft, psd=psd, modes=modes, de_workers=de_workers)

    tsft_group.file.flush()

def main():

    start_time = time.time()
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler()])

    if args.rng_seed is None:

        args.rng_seed = secrets.randbits(63)

        logger.info("No --rng_seed supplied; generated master seed %d. "
                    "Use --rng_seed %d to reproduce the random choices.",
                    args.rng_seed, args.rng_seed)

    else:
        logger.info("Using user-specified master RNG seed %d.", args.rng_seed)

    ef_rel, weight_rel = load_population(args.relaxation_file, weighted=True, args=args)
    ef_hills, _ = load_population(args.hills_file, weighted=False, args=args)

    output_dir = Path(args.repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.outfile

    rng = np.random.default_rng(args.rng_seed)
    logger.info("Initializing PSD (tdi_flag=%s).", args.tdi_flag)
    psd = PSD(args.tdi_flag)
    pool = None
    de_workers = 1

    try:

        if args.workers != 1:
            pool = mp.Pool(processes=None if args.workers == -1 else args.workers)
            de_workers = pool.map

        with h5py.File(output_path, "a") as h5f:

            results_group = h5f.require_group("results")
            existing_ids = []

            for name in results_group.keys():
                if name.startswith("run_"):
                    existing_ids.append(int(name.split("_", 1)[1]))

            next_run_idx = max(existing_ids) + 1 if existing_ids else 0
            h5f.attrs["script"] = Path(__file__).name

            if "master_rng_seed" not in h5f.attrs:
                h5f.attrs["master_rng_seed"] = int(args.rng_seed)

            h5f.attrs["latest_master_rng_seed"] = int(args.rng_seed)
            h5f.attrs["likelihood_dataset_names"] = np.asarray(["logL_ecc", "logL_circ", "lambda_LR"], dtype=h5py.string_dtype(encoding="utf-8"))
            h5f.flush()

            for offset in range(args.n_runs):

                run_index = next_run_idx + offset
                run_name = f"run_{run_index:06d}"
                run_group = results_group.create_group(run_name)
                channel, ef_inj = draw_population_sample(rng, ef_rel, weight_rel, ef_hills)
                noise_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
                modes = choose_modes(args, ef_inj, psd)
                modes_array = np.asarray(modes, dtype=int).reshape(-1, 2)

                run_group.create_dataset("channel", data=channel, dtype=h5py.string_dtype(encoding="utf-8"))
                run_group.create_dataset("ef_inj", data=float(ef_inj))
                run_group.create_dataset("noise_seed", data=noise_seed)
                run_group.create_dataset("modes", data=modes_array)
                run_group.attrs["run_index"] = int(run_index)
                run_group.attrs["master_rng_seed"] = int(args.rng_seed)

                logger.info("%s: channel=%s, e_f=%g, noise_seed=%d, modes=%s", run_name, channel, ef_inj, noise_seed, modes)
                h5f.flush()

                try:
                    for t_sft in args.vals_t_sft:
                        logger.info("%s: running T_SFT=%g s", run_name, t_sft)
                        run_one_tsft(run_group=run_group, t_sft=float(t_sft), args=args, psd=psd, modes=modes,
                                     noise_seed=noise_seed, ef_inj=ef_inj, de_workers=de_workers)
                    h5f.flush()

                except Exception:
                    logger.exception("Analysis failed for %s; partial HDF5 results were flushed.", run_name)
                    h5f.flush()
                    raise

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    logger.info("All requested runs completed. Results: %s", output_path)
    logger.info("Total script time: %.2f s", time.time() - start_time)

if __name__ == "__main__":
    main()