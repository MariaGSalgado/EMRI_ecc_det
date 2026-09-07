"""
Run the semi-coherent likelihood-ratio analysis for data containing an EMRI signal.

Examples
--------
Evaluate a range of injected eccentricities using the modes that account
for 99% of the total SNR:

    python run_det_stats.py \
        --vals_ef_inj 0 0.01 0.02 0.03 0.04 0.08 0.12 0.16 0.20 0.23 0.232 0.24 \
        --mode_selection threshold \
        --threshold 0.99 \
        --vals_t_sft 172800 \
        --outfile fid_lambda_ef_99_thr.h5

Evaluate the dependence on SFT duration:

    python run_det_stats.py \
        --vals_ef_inj 0.04 0.12 0.20 \
        --mode_selection threshold \
        --threshold 0.99 \
        --vals_t_sft 43200 86400 129600 172800 \
        --outfile fid_lambda_Tsft_99_thr.h5

Run a noisy injection multiple times using a fixed set of harmonics:

    python run_det_stats.py \
        --m1 4e6 \
        --m2 10 \
        --a 0 \
        --Tpl 0.9 \
        --vals_ef_inj 0.1 \
        --mode_selection "fixed modes" \
        --specific_modes 2,0 2,3 2,4 \
        --vals_t_sft 172800 \
        --n_runs 100 \
        --noise \
        --tdi_flag \
        --outfile pop_3_mode_ef_1e-1_noise.h5
"""

import numpy as np
import os
import time
import logging
import argparse
import h5py

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.utils.ylm import GetYlms

from utils import PSD
from inspiral import find_relevant_modes
from search import compute_lambda


traj_gen = EMRIInspiral(func=KerrEccEqFlux)
amp_gen = AmpInterpKerrEccEq()
ylm_gen = GetYlms()


print(f"PID: {os.getpid()}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Differential Evolution for EMRI likelihood ratio test to detect eccentricity.")
    # System parameters
    parser.add_argument("--m1", help="Primary mass [solar masses]", type=float, default=1e6)
    parser.add_argument("--m2", help="Secondary mass [solar masses]", type=float, default=10.0)
    parser.add_argument("--a", help="Dimensionless spin parameter", type=float, default=0.9)
    parser.add_argument("--Tpl", help="Time to plunge [years]", type=float, default=0.9)
    parser.add_argument("--xI", help="Cosine of inclination", type=float, default=1.0)
    parser.add_argument("--theta_K", help="Spin-orientation polar angle [radians]", type=float, default=np.pi/3)
    parser.add_argument("--phi_K", help="Spin-orientation azimuthal angle [radians]", type=float, default=0)

    # Search parameters
    parser.add_argument("--vals_ef_inj", help="List of final eccentricities to evaluate", nargs='+', type=float, required=True)
    parser.add_argument("--dev", help="Deviation bound for all parameters except final eccentricity", type=float, default=1e-2)
    parser.add_argument("--dev_ef", help="Deviation bound for final eccentricity", type=float, default=1e-3)
    parser.add_argument("--mode_selection", help="Mode execution strategy", choices=["fixed modes", "dominant mode", "threshold"], required=True)
    parser.add_argument("--specific_modes", help="Specific modes to use for 'fixed modes' mode selection", nargs='+', default="[2,0]")
    parser.add_argument("--threshold", help="SNR threshold for 'threshold' mode selection", type=float, default=0.99)

    parser.add_argument("--vals_t_sft", help="List of SFT durations [seconds] to scan over", nargs='+', type=float, default=172800)
    parser.add_argument("--t_data", help="Total observation time [years]", type=float, default=1.0)
    parser.add_argument("--delta_t", help="Time step [seconds]", type=float, default=5.0)
    parser.add_argument("--snr_ref", help="Reference SNR for injection", type=float, default=30.0)
    parser.add_argument("--t_snr", help="SNR reference duration [years]", type=float, default=1.0)

    # Execution
    parser.add_argument("--n_runs", help="Number of times to run the DE optimization", type=int, default=1)
    parser.add_argument("--noise", help="Whether to include noise realization", action="store_true")
    parser.add_argument("--noise_seed", help="Seed for noise realization", type=int, default=None)
    parser.add_argument("--tdi_flag", help="TDI flag for PSD", action="store_true")

    # Output
    parser.add_argument("--repo", help="Directory to save the h5py results", type=str, default="../results")
    parser.add_argument("--outfile", help="Name of the output h5 file", type=str, required=True)

    return parser.parse_args()

def main():
    start_script = time.time()
    args = parse_arguments()

    if args.mode_selection == "fixed modes":
        args.specific_modes = [
            tuple(map(int, mode.split(",")))
            for mode in args.specific_modes
        ]

    os.makedirs(args.repo, exist_ok=True)
    outpath = os.path.join(args.repo, args.outfile)

    # Initialize PSD
    logger.info("Initializing PSD...")
    psd = PSD(args.tdi_flag)

    # Default injection
    DEFAULT_INJECTION = {
        'm1': args.m1,
        'm2': args.m2,
        'a': args.a,
        'Tpl': args.Tpl,
        'ef': 0
    }

    # Base parameter bounds
    base_bounds = np.array([
        (np.log10(args.m1 * (1 - args.dev)), np.log10(args.m1 * (1 + args.dev))),
        (args.m2 * (1 - args.dev), args.m2 * (1 + args.dev)),
        (args.a * (1 - args.dev), args.a * (1 + args.dev)),
        (args.Tpl * (1 - args.dev), args.Tpl * (1 + args.dev)),
        (0, 0.9), # Will be updated in the loop
    ])
    param_names = ["m1", "m2", "a", "Tpl", "ef"]

    # Initialize the H5PY File structure
    with h5py.File(outpath, "a") as h5f:

        # Metadata
        config_grp = h5f.require_group("arguments")
        for key, val in vars(args).items():
            try:
                config_grp.attrs[key] = val
            except TypeError:
                config_grp.attrs[key] = str(val)

        results_grp = h5f.require_group("results")

        # Execution
        for local_idx in range(args.n_runs):
            logger.info(f"--- Starting local run {local_idx} ---")

            for ef_inj in args.vals_ef_inj:
                logger.info(f"Evaluating injection eccentricity: {ef_inj}")

                # Setup bounds for ef
                PARAM_BOUNDS = base_bounds.copy()
                PARAM_BOUNDS[4] = (max(0, ef_inj - args.dev_ef), ef_inj + args.dev_ef)

                # Create a group for the current ef (if it doesn't exist yet)
                ef_grp = results_grp.require_group(f"ef_{ef_inj:.6f}")

                # Determine the modes to compute based on parser arguments
                selected_modes = []
                if args.mode_selection == "fixed modes":
                    selected_modes = args.specific_modes
                elif args.mode_selection == "dominant mode":
                    all_modes, _, _, _ = find_relevant_modes(args.m1, args.m2, args.a, args.Tpl, ef_inj, args.xI, 1.0, args.theta_K, args.phi_K, psd, 0.99, traj_gen, amp_gen, ylm_gen)
                    selected_modes = [all_modes[0]]
                elif args.mode_selection == "threshold":
                    selected_modes, _, _, _ = find_relevant_modes(args.m1, args.m2, args.a, args.Tpl, ef_inj, args.xI, 1.0, args.theta_K, args.phi_K, psd, args.threshold, traj_gen, amp_gen, ylm_gen)

                logger.info(f"Using modes: {selected_modes}")

                # Save the injected ef and the selected modes at the level of the ef group (only on the first time)
                if "ef_inj" not in ef_grp:
                    ef_grp.create_dataset("ef_inj", data=ef_inj)
                if "modes" not in ef_grp:
                    modes_array = np.array(selected_modes, dtype=int)
                    ef_grp.create_dataset("modes", data=modes_array)

                # Scan over the values of T_SFT
                for t_sft in args.vals_t_sft:
                    logger.info(f" -> Running optimization for T_SFT = {t_sft}s")

                    # Create an internal subgroup to store the data relative to this T_SFT
                    tsft_name = f"tsft_{int(t_sft)}"
                    tsft_grp = ef_grp.require_group(tsft_name)

                    # Determine the next index of the run inside this subgroup
                    existing_runs = [
                        int(name.split("_")[1])
                        for name in tsft_grp.keys()
                        if name.startswith("run_")
                    ]
                    run_idx = max(existing_runs) + 1 if existing_runs else 0

                    # Differential evolution to compute the eccentricity detection statistic
                    tic = time.time()
                    logL_circ, logL_ecc, lambda_LR, h0_params, h1_params = compute_lambda(
                         ef_inj, DEFAULT_INJECTION, args.t_data, t_sft, args.delta_t,
                         args.snr_ref, args.t_snr, psd, PARAM_BOUNDS, modes=selected_modes, noise=args.noise, seed=args.noise_seed
                     )
                    tac = time.time()
                    logger.info(f"DE optimization for ef={ef_inj} and t_sft={t_sft} completed in {tac - tic:.2f} seconds.")

                    # Create a subgroup relative to the run inside the t_sft subgroup and save the data
                    run_grp = tsft_grp.create_group(f"run_{run_idx}")
                    run_grp.create_dataset("t_sft", data=t_sft)
                    run_grp.create_dataset("logL_circ", data=logL_circ)
                    run_grp.create_dataset("logL_ecc", data=logL_ecc)
                    run_grp.create_dataset("lambda_LR", data=lambda_LR)

                    h0_ds = run_grp.create_dataset("h0_params", data=h0_params)
                    h0_ds.attrs["param_names"] = param_names

                    h1_ds = run_grp.create_dataset("h1_params", data=h1_params)
                    h1_ds.attrs["param_names"] = param_names

                    h5f.flush()
                    logger.info(f"Flushed: ef={ef_inj}, t_sft={t_sft}, run={run_idx}")

    logger.info(f"All runs completed. Data safely archived at {outpath}")
    logger.info(f"Total Script Time: {time.time() - start_script:.2f} seconds")


if __name__ == "__main__":
    main()