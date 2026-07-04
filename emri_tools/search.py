import numpy as np
import scipy as sp
from multiprocess import Pool, cpu_count

from gw_signal import get_f_fdot_fddot_back, generate_injection_data


def dirichlet_kernel(f_0, T_sft):
    """The physically correct kernel for the f_1 = 0 case."""
    return T_sft * np.exp(1j * np.pi * f_0 * T_sft) * np.sinc(f_0 * T_sft)


def fresnel_kernel(f_0, f_1, T_sft):
    """
    Fresnel kernel.

    Args:
        f_0 (float or np.ndarray): Initial frequency.
        f_1 (float or np.ndarray): Frequency derivative.
        T_sft (float): Duration of the SFT segment.

    Returns:
        np.ndarray: Fresnel kernel values.
    """

    factor = np.sqrt(2 * f_1)

    eps = 1e-12
    f_1_safe = np.where(np.abs(f_1) < eps, eps, f_1)
    factor_safe = np.where(np.abs(factor) < eps, eps, factor)

    quot = f_0 / f_1_safe

    arg1 = factor_safe * quot
    arg2 = factor_safe * (quot + T_sft)

    arg1 = np.clip(arg1, -50, 50)
    arg2 = np.clip(arg2, -50, 50)

    Sl, Cl = sp.special.fresnel(arg1)
    Su, Cu = sp.special.fresnel(arg2)

    phase = np.exp(-1j * np.pi * f_0 ** 2 / f_1_safe)

    kernel = phase * ((Cu - Cl) + 1j * (Su - Sl)) / factor_safe

    return np.where(np.isfinite(kernel), kernel, 0.0)


def robust_kernel(f_0, f_1, T_sft):
    """
    This is the function we will call. It computes the forward pass.
    """
    abs_f_1 = np.abs(f_1)
    return np.where(
        abs_f_1 > 0.,
        np.where(
            f_1 > 0,
            fresnel_kernel(f_0, abs_f_1, T_sft),
            fresnel_kernel(-f_0, abs_f_1, T_sft).conj(),
        ),
        dirichlet_kernel(f_0, T_sft),
    )


def det_stat(data_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, psd, P=100, T_sft=86400.):
    """
    Compute the detection statistic for a given set of parameters and data.
    This function calculates a detection statistic based on the input Short Fourier Transforms (SFTs)
    and signal parameters. It evaluates the likelihood of a signal being present in the data.
    Args:
        data_sfts (jax.numpy.ndarray): The input SFT data, typically a 2D array where the first
            dimension corresponds to frequency bins and the second dimension corresponds to time segments.
        A_alpha (jax.numpy.ndarray): Amplitude parameters of the signal.
        phi_alpha (jax.numpy.ndarray): Phase parameters of the signal.
        f_alpha (jax.numpy.ndarray): Frequency parameters of the signal.
        fdot_alpha (jax.numpy.ndarray): Frequency derivative parameters of the signal.
        P (int, optional): The range of frequency bins to consider around the signal frequency.
            Defaults to 100.
        T_sft (float, optional): The duration of each SFT segment in seconds. Defaults to 86400 (1 day).
    Returns:
        float: The detection statistic value, normalized by the square root of the noise-weighted
        signal power.
    Notes:
        - The function uses Fresnel kernels to compute the signal contribution in the frequency domain.
        - Out-of-range frequency bins are masked to zero to avoid indexing errors.
        - The detection statistic is computed as the ratio of the matched-filter term (`dh_term`)
          to the square root of the noise-weighted signal power (`hh_term`).
    """

    # Non-signal-dependent values are passed here by clousure
    deltaf = 1 / T_sft

    f_k_of_alpha = (f_alpha * T_sft).astype(int)
    k_min_max = f_k_of_alpha + np.arange(-P, P + 1)[:, None]
    k_min_max_clipped = np.clip(k_min_max, 0, data_sfts.shape[0] - 1)

    # Set to 0 whatever gets beyond the range.
    # Note that jax will not complain about out-of-range indexing
    zero_mask = (k_min_max >= 0) & (k_min_max < data_sfts.shape[0])

    c_alpha = (
            deltaf
            * data_sfts[k_min_max_clipped, np.arange(data_sfts.shape[1])].conj() / psd(k_min_max * deltaf)
            * robust_kernel(f_alpha - k_min_max * deltaf, fdot_alpha, T_sft)
            * zero_mask
    ).sum(axis=0)

    c_alpha = np.nan_to_num(c_alpha, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    dh_term = 2 * A_alpha * c_alpha * np.exp(1j * phi_alpha)
    hh_term = T_sft * (A_alpha ** 2 / psd(f_alpha)) / 2
    return dh_term, hh_term


def compute_log_likelihood(params, t_alpha, data_sfts, T_SFT, psd, modes=[(2, 0)], delta_phi_max=1.0):
    """
    Compute log-likelihood for EMRI detection statistic considering multiple modes.
    """

    # Convert parameters to physical values
    physical_params = np.array([
        10 ** params[0],  # m1 from log10(m1)
        params[1],  # m2
        params[2],  # a (spin)
        params[3],  # Tpl
        params[4],  # ef
        1.0  # x0
    ])

    try:
        # Get the orbital phases, fundamental frequencies, and their derivatives
        phi, f, dotf, dotdotf = get_f_fdot_fddot_back(physical_params, t_alpha, err=1e-6)
    except Exception:
        return -1e50  # Return very low likelihood for failed waveform generation

    log_like_total = 0.0

    for (m, n) in modes:
        # Compute frequency track for harmonic (m, n)
        phi_alpha = m * phi[0] + n * phi[1]
        f_alpha = m * f[0] + n * f[1]
        fdot_alpha = m * dotf[0] + n * dotf[1]
        fddot_alpha = m * dotdotf[0] + n * dotdotf[1]

        # Phase coherence condition
        delta_phi_approx = np.abs(fddot_alpha * T_SFT ** 3 / 6)

        # Amplitude mask: valid frequency bins
        A_alpha = np.where(
            (f_alpha > 10 ** (-3.5)) &
            (delta_phi_approx < delta_phi_max),
            1.0, 0.0
        )

        # Compute detection statistic
        dh, hh = det_stat(data_sfts, A_alpha, phi_alpha, f_alpha, fdot_alpha, psd, T_sft=T_SFT)

        # Sum detection statistic over valid bins and add it to the total log-likelihood
        log_like_total += 0.5 * np.sum(A_alpha * np.divide(
            np.abs(dh) ** 2, hh, where=hh > 1e-20, out=np.zeros_like(hh)
        ))

    if not np.isfinite(log_like_total):
        return -1e100

    return log_like_total


def compute_lambda(ef_inj, DEFAULT_INJECTION, T_DATA, T_SFT, DELTA_T, SNR_REF, T_SNR, psd, PARAM_BOUNDS, modes=[(2, 0)], noise=True, seed=42):

    # Generate data with the specified final eccentricity

    INJECTION = DEFAULT_INJECTION.copy()
    INJECTION["ef"] = ef_inj
    all_data = generate_injection_data(INJECTION, T_DATA, T_SFT, DELTA_T, SNR_REF, T_SNR, psd, noise, seed)
    data_sfts = all_data['data_sfts']
    t_alpha = all_data['t_alpha']

    # Find the maximum of the likelihood allowing eccentricity

    with Pool(processes=cpu_count()) as pool:
        result = sp.optimize.differential_evolution(
            lambda params: -compute_log_likelihood(params, t_alpha, data_sfts, T_SFT, psd, modes),
            bounds=PARAM_BOUNDS,
            strategy='randtobest1bin',
            maxiter=500,
            popsize=20,
            tol=1e-3,
            mutation=(0.5, 1.5),
            recombination=0.5,
            seed=42,
            workers=pool.map,
            disp=True,
            polish=True
        )

    logL_ecc = - result.fun
    best_params_ecc = result.x

    # Find the maximum of the likelihood assuming zero eccentricity

    with Pool(processes=cpu_count()) as pool:
        result = sp.optimize.differential_evolution(
            lambda params: -compute_log_likelihood(list(params) + [0.0], t_alpha, data_sfts, T_SFT, psd, modes),
            bounds=PARAM_BOUNDS[:4],
            strategy='randtobest1bin',
            maxiter=500,
            popsize=20,
            tol=1e-3,
            mutation=(0.5, 1.5),
            recombination=0.5,
            seed=42,
            workers=pool.map,
            disp=True,
            polish=True
        )

    logL_circ = - result.fun
    best_params_circ = result.x

    lambda_LR = 2 * (logL_ecc - logL_circ)

    return logL_circ, logL_ecc, lambda_LR, best_params_circ, best_params_ecc