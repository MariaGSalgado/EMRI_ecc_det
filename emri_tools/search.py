import numpy as np
import scipy as sp
from multiprocess import Pool, cpu_count
from functools import partial

from gw_signal import get_f_fdot_fddot_back, generate_injection_data


def dirichlet_kernel(f_0, T_sft):
    """
    The physically correct kernel for the f_1 = 0 case.
    """

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
    Computes the forward pass.

    Args:
        f_0 (float or np.ndarray): Initial frequency.
        f_1 (float or np.ndarray): Frequency derivative.
        T_sft (float): Duration of the SFT segment.

    Returns:
        np.ndarray: Robust kernel values.
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
    Computes the detection statistic for a given set of parameters and data.
    This function calculates a detection statistic based on the input Short Fourier Transforms (SFTs)
    and signal parameters. It evaluates the likelihood of a signal being present in the data.

    Args:
        data_sfts (np.ndarray): The input SFT data, typically a 2D array where the first
            dimension corresponds to frequency bins and the second dimension corresponds to time segments.
        A_alpha (np.ndarray): Amplitude parameters of the signal.
        phi_alpha (np.ndarray): Phase parameters of the signal.
        f_alpha (np.ndarray): Frequency parameters of the signal.
        fdot_alpha (np.ndarray): Frequency derivative parameters of the signal.
        psd (callable): Function that returns PSD values for given frequencies.
        P (int, optional): The range of frequency bins to consider around the signal frequency (default: 100).
        T_sft (float, optional): The duration of each SFT segment in seconds (default: 86400 (1 day)).

    Returns:
        dh_term (np.ndarray): Noise-weighted data-signal inner-product term
            evaluated for each SFT.
        hh_term (np.ndarray): Noise-weighted signal self-inner-product term
            evaluated for each SFT.

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
    dh_term = 4 * A_alpha * c_alpha * np.exp(1j * phi_alpha)
    hh_term = 4 * T_sft * (A_alpha ** 2 / psd(f_alpha))

    return dh_term, hh_term


def compute_log_likelihood(params, t_alpha, data_sfts, T_SFT, psd, modes=[(2, 0)], delta_phi_max=1.0):
    """
    Computes the log-likelihood for EMRI detection statistic considering multiple modes.

    Args:
        params (array-like): EMRI parameters in the inference parameterization:
            [log10(m1), m2, a, Tpl, ef].
        t_alpha (np.ndarray): Array of SFT time samples in seconds.
        data_sfts (np.ndarray): Complex SFT data.
        T_SFT (float): Duration of each SFT segment in seconds.
        psd (callable): Function that returns PSD values for given frequencies.
        modes (list of tuple, optional): List of (m, n) harmonic indices to
            include in the likelihood (default: [(2, 0)]).
        delta_phi_max (float, optional): Maximum allowed phase error due to
            the second-order frequency evolution within an SFT (default: 1 radian).

    Returns:
        float: Log-likelihood value for the supplied parameter set.
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
    

def _negative_log_likelihood(params, t_alpha, data_sfts, T_SFT, psd, modes, fixed_ef=None):
    if fixed_ef is not None:
        params = np.append(params, fixed_ef)
    return -compute_log_likelihood(params, t_alpha, data_sfts, T_SFT, psd, modes)


def _run_differential_evolution(objective, bounds, workers, init="sobol", seed=42):
    options = {
        "strategy": "randtobest1bin",
        "maxiter": 300,
        "popsize": 15,
        "tol": 1e-3,
        "mutation": (0.5, 1.5),
        "recombination": 0.5,
        "seed": seed,
        "init": init,
        "disp": True,
        "polish": True
        }

    if workers != 1:
        options["updating"] = "deferred"
    return sp.optimize.differential_evolution(objective, bounds=bounds, workers=workers, **options)


def compute_lambda(ef_inj, DEFAULT_INJECTION, T_DATA, T_SFT, DELTA_T, SNR_REF, T_SNR, psd, PARAM_BOUNDS, modes=[(2, 0)], noise=False, seed=42, workers=8):
    """
    Computes the likelihood-ratio detection statistic for eccentricity.

    Args:
        ef_inj (float): Final eccentricity of the injected EMRI signal.
        DEFAULT_INJECTION (dict): Dictionary containing the default physical
            parameters of the EMRI injection.
        T_DATA (float): Duration of the simulated data in years.
        T_SFT (float): Duration of each SFT segment in seconds.
        DELTA_T (float): Sampling interval of the time-domain waveform in seconds.
        SNR_REF (float): Target SNR to which the injected signal is scaled.
        T_SNR (float): Duration of the waveform used to determine the reference SNR, in years.
        psd (callable): Function that returns PSD values for given frequencies.
        PARAM_BOUNDS (sequence): Parameter bounds used for the likelihood maximization.
        modes (list of tuple, optional): List of (m, n) harmonic indices included in the likelihood (default: [(2, 0)]).
        noise (bool, optional): If True, adds a noise realization to the injected signal (default: True).
        seed (int, optional): Random seed used for the noise generation (default: 42).
        workers (int or callable, optional): Worker count or map-like callable for differential evolution (default: 8).

    Returns:
        logL_circ (float): Maximum log-likelihood under the circular hypothesis.
        logL_ecc (float): Maximum log-likelihood under the eccentric hypothesis.
        lambda_LR (float): Likelihood-ratio detection statistic.
        best_params_circ (np.ndarray): Best-fit parameters under the circular hypothesis.
        best_params_ecc (np.ndarray): Best-fit parameters under the eccentric hypothesis.
    """

    # Generate data with the specified final eccentricity

    INJECTION = DEFAULT_INJECTION.copy()
    INJECTION["ef"] = ef_inj
    all_data = generate_injection_data(INJECTION, T_DATA, T_SFT, DELTA_T, SNR_REF, T_SNR, psd, noise, seed)
    data_sfts = all_data['data_sfts']
    t_alpha = all_data['t_alpha']

    objective = partial(
        _negative_log_likelihood,
        t_alpha=t_alpha,
        data_sfts=data_sfts,
        T_SFT=T_SFT,
        psd=psd,
        modes=modes
    )

    # Find the maximum of the likelihood allowing eccentricity

    result = _run_differential_evolution(objective, PARAM_BOUNDS, workers, seed=seed)

    logL_ecc = - result.fun
    best_params_ecc = result.x

    # Find the maximum of the likelihood assuming zero eccentricity

    circular_objective = partial(objective, fixed_ef=0.0)
    result = _run_differential_evolution(circular_objective, PARAM_BOUNDS[:4], workers, seed=seed)

    logL_circ = - result.fun
    best_params_circ = result.x

    lambda_LR = 2 * (logL_ecc - logL_circ)

    return logL_circ, logL_ecc, lambda_LR, best_params_circ, best_params_ecc