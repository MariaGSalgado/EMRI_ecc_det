import numpy as np
import scipy as sp
import pandas as pd
import h5py
import os


def load_hdf5(filepath):
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


def load_psd(tdi_flag):
    """
    Load and return a Power Spectral Density (PSD) as a cubic spline.

    Args:
        tdi_flag (bool): If True, use the TDI PSD. Otherwise, use the standard LISA PSD.

    Returns:
        cs_eval (scipy.interpolate.CubicSpline): A callable spline function representing the PSD.
            Can be evaluated as `cs_eval(frequencies)` to get the PSD at the desired frequencies.
    """

    base_dir = os.path.dirname(__file__)

    if tdi_flag:
        psd_file = os.path.join(base_dir, 'TDI2_AE_psd.npy')
        ff_tdi, Sn_tdi = np.load(psd_file).T
        Sn_tdi = np.clip(Sn_tdi, Sn_tdi[0], None)
        cs_eval = sp.interpolate.CubicSpline(ff_tdi, Sn_tdi)
        print("Using TDI PSD")

    else:
        psd_file = os.path.join(base_dir, 'LPA.txt')
        ff, Sn = np.loadtxt(psd_file,skiprows=1).T
        cs_eval = sp.interpolate.CubicSpline(ff, Sn**2)
        print("Using LISA PSD")

    return cs_eval


class PSD:
    def __init__(self, tdi_flag=False):
        self.cs_eval = load_psd(tdi_flag)

    def __call__(self, frequencies):
        ret = self.cs_eval(np.abs(frequencies))
        mask = (np.abs(frequencies) < 1e-4)
        ret[mask] = self.cs_eval(1e-4)
        return ret


def compute_sfts(data, delta_T, wind, samples_per_sft):
    """
    Compute SFTs for the given data.

    Args:
        data (np.ndarray): Input time-domain data.
        delta_T (float): Time step in seconds.
        wind (np.ndarray): Window function of length samples_per_sft.
        samples_per_sft (int): Number of samples per SFT.

    Returns:
        np.ndarray: SFTs with shape (num_freq_bins, num_sfts).
    """
    num_sfts = data.size // samples_per_sft
    sfts = (
        delta_T
        * np.fft.rfft(
            wind * data[: num_sfts * samples_per_sft].reshape(-1, samples_per_sft), axis=1
        ).T
    )
    return sfts


def sft_inner_product(sft1, sft2, freq, psd):
    """
    Compute the inner product of two SFTs in the frequency domain.

    Args:
        sft1 (np.ndarray): First SFT.
        sft2 (np.ndarray): Second SFT.
        freq (np.ndarray): Frequency bins corresponding to the SFTs.

    Returns:
        np.ndarray: Inner product of the two SFTs.
    """
    df = freq[1] - freq[0]
    return 4 * np.sum(sft1.conj() * sft2 / psd(freq)[:, None] * df, axis=0).real


def get_snr(signal, delta_T, psd):
    """
    Compute the signal-to-noise ratio (SNR) of a time-domain signal.

    Args:
        signal (np.ndarray): Time-domain signal array.
        delta_T (float): Time step between consecutive samples (seconds).

    Returns:
        float: Signal-to-noise ratio (SNR) of the input signal.
    """
    ft_freq = np.fft.rfftfreq(len(signal), delta_T)
    SNR2_source = 4 * (np.abs(np.fft.rfft(signal)*delta_T)**2 / psd(np.abs(ft_freq))).sum() * np.diff(ft_freq)[0]
    return SNR2_source**0.5