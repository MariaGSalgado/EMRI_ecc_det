import numpy as np
import pandas as pd
import h5py
import os
import shutil
from pathlib import Path

def load_hdf5(filepath):
    """
    Loads likelihood and parameter-estimation results from an HDF5 file.

    Args:
        filepath (str): Path to the HDF5 file containing the results.

    Returns:
        pd.DataFrame: DataFrame containing one row per run. Columns include
            the injected eccentricity, SFT duration, run number, selected
            modes, circular and eccentric maximum log-likelihoods, the
            likelihood-ratio statistic, and the best-fit parameters under
            both hypotheses.
    """

    rows = []

    with h5py.File(filepath, "r") as h5f:

        results = h5f["results"]

        for ef_key in results.keys():

            ef_grp = results[ef_key]

            ef_val = float(ef_grp["ef_inj"][()])
            modes = ef_grp["modes"][()].tolist()

            for tsft_key in ef_grp.keys():

                if not tsft_key.startswith("tsft_"):
                    continue

                tsft_grp = ef_grp[tsft_key]
                t_sft = int(tsft_key.split("_")[1])

                for run_key in tsft_grp.keys():

                    if not run_key.startswith("run_"):
                        continue

                    run_grp = tsft_grp[run_key]

                    param_names = list(run_grp["h0_params"].attrs["param_names"])
                    h0 = run_grp["h0_params"][()]
                    h1 = run_grp["h1_params"][()]

                    row = {
                        "ef_inj": ef_val,
                        "t_sft": t_sft,
                        "run": int(run_key.split("_")[1]),
                        "modes": str(modes),
                        "logL_circ": float(run_grp["logL_circ"][()]),
                        "logL_ecc": float(run_grp["logL_ecc"][()]),
                        "lambda_LR": float(run_grp["lambda_LR"][()]),
                    }

                    for pname, v0, v1 in zip(param_names, h0, h1):
                        row[f"h0_{pname}"] = v0
                        row[f"h1_{pname}"] = v1
                    row[f"h1_ef"] = h1[-1]

                    rows.append(row)

    return pd.DataFrame(rows)


def load_hdf5_pop(filepath):
    """
    Load EMRI population results into a wide pandas DataFrame.

    One row per realization and T_SFT.
    Broad and narrow results are stored in separate columns.

    Includes:
      - Injected source parameters
      - Log-likelihoods and likelihood-ratio statistic
      - Best-fit parameters for eccentric and circular models
    """

    def decode(value):
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        return value

    def sort_key(name, prefix):
        suffix = name[len(prefix):]
        return (0, int(suffix)) if suffix.isdigit() else (1, name)

    default_param_names = [
        "log10_m1", "m2", "a", "Tpl", "ef"
    ]

    rows = []

    with h5py.File(filepath, "r") as h5f:
        if "results" not in h5f:
            raise KeyError(f"No 'results' group found in {filepath}")

        results = h5f["results"]

        run_names = sorted(
            (
                name for name in results
                if name.startswith("run_")
                and isinstance(results[name], h5py.Group)
            ),
            key=lambda name: sort_key(name, "run_"),
        )

        for run_name in run_names:
            run = results[run_name]

            run_suffix = run_name.removeprefix("run_")
            run_index = run.attrs.get(
                "run_index",
                int(run_suffix) if run_suffix.isdigit() else np.nan,
            )

            channel = (
                decode(run["channel"][()])
                if "channel" in run else None
            )
            ef_inj = (
                float(run["ef_inj"][()])
                if "ef_inj" in run else np.nan
            )
            noise_seed = (
                int(run["noise_seed"][()])
                if "noise_seed" in run else np.nan
            )
            master_seed = run.attrs.get(
                "master_rng_seed",
                h5f.attrs.get("master_rng_seed", np.nan),
            )

            tsft_names = sorted(
                (
                    name for name in run
                    if name.startswith("tsft_")
                    and isinstance(run[name], h5py.Group)
                ),
                key=lambda name: sort_key(name, "tsft_"),
            )

            for tsft_name in tsft_names:
                tsft = run[tsft_name]

                row = {
                    "run_index": (
                        int(run_index)
                        if np.isfinite(run_index) else np.nan
                    ),
                    "channel": channel,
                    "ef_inj": ef_inj,
                    "noise_seed": noise_seed,
                    "master_rng_seed": decode(master_seed),
                    "t_sft": (
                        float(tsft["t_sft"][()])
                        if "t_sft" in tsft else np.nan
                    ),
                }

                # Injected source parameters
                inj = tsft.get("injection_parameters")
                if inj is not None and "base_source_parameters" in inj:
                    source_ds = inj["base_source_parameters"]
                    source_values = np.asarray(
                        source_ds[...], dtype=float
                    )

                    names = source_ds.attrs.get(
                        "param_names",
                        ["m1", "m2", "a", "Tpl", "ef", "xI"],
                    )
                    names = [str(decode(name)) for name in names]

                    if source_values.ndim == 2 and len(source_values) > 0:
                        for name, value in zip(names, source_values[0]):
                            if name != "ef":
                                row[f"{name}_inj"] = float(value)

                # Broad and narrow prior groups
                prior_groups = (
                    ("broad", tsft),
                    ("narrow", tsft.get("narrow_bounds")),
                )

                data_names = (
                    ("eccentric_signal", "ecc_signal"),
                    ("circular_signal", "circ_signal"),
                    ("noise_only", "noise"),
                )

                for prior_name, prior_group in prior_groups:
                    if prior_group is None:
                        continue

                    for h5_data_name, column_data_name in data_names:
                        if h5_data_name not in prior_group:
                            continue

                        data_group = prior_group[h5_data_name]
                        prefix = f"{prior_name}_{column_data_name}"

                        # Log-likelihood values
                        for key in ("logL_ecc", "logL_circ"):
                            if key in data_group:
                                row[f"{prefix}_{key}"] = float(
                                    data_group[key][()]
                                )

                        # Likelihood-ratio statistic
                        if "lambda_LR" in data_group:
                            row[f"{prefix}_lambda_LR"] = float(
                                data_group["lambda_LR"][()]
                            )
                        elif (
                            "logL_ecc" in data_group
                            and "logL_circ" in data_group
                        ):
                            row[f"{prefix}_lambda_LR"] = 2.0 * (
                                float(data_group["logL_ecc"][()])
                                - float(data_group["logL_circ"][()])
                            )
                            
                        # Best-fit parameters for each model
                        for model, fit_name in (
                            ("ecc", "fit_eccentric"),
                            ("circ", "fit_circular"),
                        ):

                            fit_group = data_group[fit_name]

                            best_ds = fit_group["best_params"]
                            best_params = np.asarray(
                                best_ds[...], dtype=float
                            ).reshape(-1)

                            param_names = best_ds.attrs.get(
                                "param_names",
                                default_param_names,
                            )
                            param_names = [
                                str(decode(name))
                                for name in param_names
                            ]


                            for param_name, value in zip(
                                param_names, best_params
                            ):
                                row[
                                    f"{prefix}_{model}_{param_name}"
                                ] = float(value)

                            if "log10_m1" in param_names:
                                idx = param_names.index("log10_m1")
                                row[f"{prefix}_{model}_m1"] = (
                                    10.0 ** best_params[idx]
                                )

                rows.append(row)

    return pd.DataFrame(rows)


def merge_hdf5(output_file, *input_files):

    if len(input_files) == 0:
        raise ValueError("At least one input file is required.")

    shutil.copy(input_files[0], output_file)

    with h5py.File(output_file, "a") as dst:

        dst_results = dst.require_group("results")

        for source_file in input_files[1:]:

            with h5py.File(source_file, "r") as src:

                src_results = src["results"]

                for ef_name in src_results.keys():

                    src_ef = src_results[ef_name]
                    dst_ef = dst_results.require_group(ef_name)

                    for ds_name in ["ef_inj", "modes"]:

                        if ds_name not in dst_ef and ds_name in src_ef:
                            src.copy(src_ef[ds_name], dst_ef, name=ds_name)

                    for tsft_name in src_ef.keys():

                        if not tsft_name.startswith("tsft_"):
                            continue

                        src_tsft = src_ef[tsft_name]
                        dst_tsft = dst_ef.require_group(tsft_name)

                        existing_runs = [
                            int(name.split("_")[1])
                            for name in dst_tsft.keys()
                            if name.startswith("run_")
                        ]

                        next_run = max(existing_runs) + 1 if existing_runs else 0

                        for run_name in src_tsft.keys():

                            src_run = src_tsft[run_name]

                            new_run_name = f"run_{next_run}"
                            next_run += 1

                            src.copy(src_run, dst_tsft, name=new_run_name)

    print(f"Merged files into '{output_file}'")


def merge_hdf5_pop(output_file, *input_files):

    if len(input_files) == 0:
        raise ValueError("At least one input file is required.")

    # Prevent overwriting an input file.
    output_path = os.path.realpath(output_file)
    input_paths = [os.path.realpath(path) for path in input_files]

    if output_path in input_paths:
        raise ValueError(
            "output_file must be different from every input file."
        )

    # Create the output directory if necessary.
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    # Copy the first file, preserving its complete HDF5 structure.
    shutil.copy(input_files[0], output_file)

    with h5py.File(output_file, "a") as dst:

        dst_results = dst.require_group("results")

        # Find the next available run index.
        existing_runs = [
            int(name.split("_", 1)[1])
            for name in dst_results.keys()
            if name.startswith("run_")
            and name.split("_", 1)[1].isdigit()
        ]

        next_run = max(existing_runs) + 1 if existing_runs else 0

        # Append runs from the remaining input files.
        for source_file in input_files[1:]:

            with h5py.File(source_file, "r") as src:

                if "results" not in src:
                    raise KeyError(
                        f"No 'results' group found in {source_file}"
                    )

                src_results = src["results"]

                for run_name in sorted(src_results.keys()):

                    if not run_name.startswith("run_"):
                        continue

                    suffix = run_name.split("_", 1)[1]
                    if not suffix.isdigit():
                        continue

                    src_run = src_results[run_name]
                    new_run_name = f"run_{next_run:06d}"

                    src.copy(
                        src_run,
                        dst_results,
                        name=new_run_name,
                    )

                    next_run += 1

        dst.flush()

    print(f"Merged {len(input_files)} files into '{output_file}'")


def load_population(filename, weighted, args=None, ef_min=0, ef_max=1):
    """Load a population CSV, retain valid epl rows, and normalize weights if needed."""

    results_dir = Path(__file__).resolve().parent.parent / "populations"
    filepath = results_dir / filename

    df = pd.read_csv(filepath)
    ef = pd.to_numeric(df["epl"], errors="coerce").to_numpy(dtype=float)
    if args is not None:
        ef_min, ef_max = map(float, args.ef_bounds)
    valid = np.isfinite(ef) & (ef >= ef_min) & (ef <= ef_max)
    weights = None

    if weighted:
        weights = pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)
        valid &= np.isfinite(weights) & (weights >= 0)

    ef = ef[valid]

    if weights is not None:
        weights = weights[valid]

    order = np.argsort(ef)
    ef = ef[order]

    if weights is not None:
        weights = weights[order]
        total_weight = float(np.sum(weights))
        weights = weights / total_weight

    return ef, weights
