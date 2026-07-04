import numpy as np
import scipy as sp

import few
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.waveform import GenerateEMRIWaveform
from few.utils.geodesic import get_separatrix
from few.utils.constants import YRSID_SI
from lisatools.detector import EqualArmlengthOrbits
from fastlisaresponse import ResponseWrapper

from utils import get_snr, compute_sfts, sft_inner_product


traj_gen = EMRIInspiral(func=KerrEccEqFlux)
rhs = KerrEccEqFlux()
wave_gen = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs = dict(pad_output=True),
    return_list=False
)


def get_f_fdot_fddot_back(params, t_alpha, err=1e-11):
    """
    Compute the phase, frequency, and their first and second derivatives for a backward-integrated inspiral trajectory.
    This function integrates the inspiral trajectory backwards in time, starting from the separatrix, and evaluates the orbital phase,
    frequency, and their derivatives at specified time samples. The results are returned for two harmonics (indices 3 and 5 in the
    trajectory output).
    Parameters
    ----------
    params : np.ndarray
        Array of shape (N, 6) or (6,) containing the physical parameters for the inspiral:
        [m1, m2, a, Tpl, e0, x0], where
            m1 : float
                Mass of the primary object.
            m2 : float
                Mass of the secondary object.
            a : float
                Spin parameter.
            Tpl : float
                Plunge time (in years).
            e0 : float
                Initial eccentricity.
            x0 : float
                Initial semi-latus rectum or related parameter.
    t_alpha : np.ndarray
        Array of time samples (in seconds) at which to evaluate the phase and frequency quantities.
    Returns
    -------
    phi : np.ndarray
            Array of shape (2, len(t_alpha)) containing the orbital phases for the two harmonics at each time sample.
        f : np.ndarray
            Array of shape (2, len(t_alpha)) containing the frequencies for the two harmonics at each time sample.
        dotf : np.ndarray
            Array of shape (2, len(t_alpha)) containing the first derivatives of the frequencies (frequency derivatives) for the two harmonics.
        dotdotf : np.ndarray
            Array of shape (2, len(t_alpha)) containing the second derivatives of the frequencies for the two harmonics.
        Notes
        -----
        - The function assumes the existence of global objects and methods such as `traj_gen`, `rhs`, `get_separatrix`, and constants like `YRSID_SI`.
        - Only the time samples within the valid integration range are populated; others remain zero.
        - The function is tailored for EMRI (Extreme Mass Ratio Inspiral) waveform modeling and relies on external trajectory and spline evaluation routines.
        """

    # Define ODE flux and parameters
    m1, m2, a, Tpl, ef, x0 = params.T
    mass_ratio = m1 * m2 / (m1 + m2) ** 2
    rhs.add_fixed_parameters(params[0], params[1], a)

    # Define max time based on t_alpha
    p_0 = traj_gen.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, ef, x0) + 1e-6
    backwards_result = traj_gen(m1, m2, a, p_0, ef, x0, T=Tpl, integrate_backwards=True, err=err)

    t_int = traj_gen.integrator_spline_t

    # New t
    new_t_alpha = (t_alpha + YRSID_SI * Tpl - t_alpha[-1])[::-1]
    mask = (new_t_alpha > 0.0) & (new_t_alpha < t_int.max())

    shape_alpha = t_alpha.shape[0]
    phi = np.zeros((2, shape_alpha))
    f = np.zeros((2, shape_alpha))
    dotf = np.zeros((2, shape_alpha))
    dotdotf = np.zeros((2, shape_alpha))

    # Time
    t_adim = t_int * mass_ratio
    t_alpha_adim = new_t_alpha[mask] * mass_ratio

    # Phases
    phi_ = traj_gen.inspiral_generator.eval_integrator_spline(new_t_alpha[mask])
    phi[0][mask] = phi_[:, 3]
    phi[1][mask] = phi_[:, 5]

    # Frequencies
    f_ = traj_gen.inspiral_generator.eval_integrator_derivative_spline(new_t_alpha[mask], order=1)
    f[0][mask] = -f_[:, 3] / (np.pi * 2)
    f[1][mask] = -f_[:, 5] / (np.pi * 2)

    # Derivatives
    dotf_ = traj_gen.inspiral_generator.eval_integrator_derivative_spline(new_t_alpha[mask], order=2)
    dotf[0][mask] = dotf_[:, 3] / (np.pi * 2)
    dotf[1][mask] = dotf_[:, 5] / (np.pi * 2)

    # Second derivatives
    dotdotf_ = traj_gen.inspiral_generator.eval_integrator_derivative_spline(new_t_alpha[mask], order=3)
    dotdotf[0][mask] = -dotdotf_[:, 3] / (np.pi * 2)
    dotdotf[1][mask] = -dotdotf_[:, 5] / (np.pi * 2)

    return phi, f, dotf, dotdotf


def get_frequency_evolution(true_values, T_data, T_sft, delta_T):
    """Compute the frequency evolution of an EMRI signal."""

    # Generate EMRI signal (only to obtain time array)
    t, hp, hx = create_signal(true_values, T=T_data, delta_T=delta_T)

    # Trim signal to desired duration
    mask = t < YRSID_SI * T_data
    t = t[mask]
    hp = hp[mask]
    hx = hx[mask]

    # Instantaneous amplitude
    amp_td = np.sqrt(hp**2 + hx**2)

    # SFT structure
    samples_per_sft = int(T_sft / delta_T)
    num_sfts = len(t) // samples_per_sft

    # Truncate to full SFTs only
    n_total = num_sfts * samples_per_sft

    t = t[:n_total]
    amp_td = amp_td[:n_total]

    # SFT times
    t_alpha = np.arange(num_sfts) * T_sft

    # Reshape into SFT blocks
    amp_blocks = amp_td.reshape(num_sfts, samples_per_sft)

    # Mean amplitude per SFT
    amp_alpha = np.mean(amp_blocks, axis=1)

    # Get frequency evolution
    phi_f_fdot_fddot = get_f_fdot_fddot_back(true_values, t_alpha)

    return t_alpha, phi_f_fdot_fddot, amp_alpha


def get_response_generator(T, dt):
    """Function to obtain the waveform generator with LISA response.

    Args:
        T (float): Observation time in years
        dt (float): time step in seconds

    Returns:
        func: Function that takes in the parameters and returns the waveform
    """

    use_gpu = few.has_backend('gpu')
    tdi_gen = "2nd generation"
    tdi_kwargs_esa = dict(
        t0=30000.0,
        order=25,
        tdi=tdi_gen,
        tdi_chan="AET",
        orbits=EqualArmlengthOrbits(),
    )

    # With longer signals, we care less about this
    t0 = 10000.0  # Throw away on both ends when our orbital information is weird
    wave_gen_resp = ResponseWrapper(
        wave_gen,
        T,
        dt,
        index_lambda=8,
        index_beta=7,
        flip_hx=True,  # Set to True if waveform is h+ - ihx (FEW is)
        is_ecliptic_latitude=False,  # False if using polar angle (theta)
        remove_garbage='zero',  # Removes the beginning of the signal that has bad information
        **tdi_kwargs_esa,
    )
    return wave_gen_resp

def create_signal(params,
                  dist=1.0,
                  qS=np.pi / 3, phiS=0.0,
                  qK=np.pi / 3, phiK=0.0,
                  Phi_phi0=0.0, Phi_theta0=0.0, Phi_r0=0.0,
                  T=1.0, delta_T=10.0, eps=1e-2,
                  return_params=False,
                  randomize=False,
                  resp_gen=True):
    """
    Generates a gravitational wave signal for an EMRI (Extreme Mass Ratio Inspiral) system using the provided parameters.

    Parameters
    ----------
    params : array-like
        Array or object containing the EMRI system parameters (m1, m2, a, Tpl, ef, x0).
    dist : float, optional
        Distance to the source (default: 1.0).
    qS : float, optional
        Source sky position polar angle (default: np.pi/3).
    phiS : float, optional
        Source sky position azimuthal angle (default: 0.0).
    qK : float, optional
        Spin orientation polar angle (default: np.pi/3).
    phiK : float, optional
        Spin orientation azimuthal angle (default: 0.0).
    Phi_phi0 : float, optional
        Initial phase in the phi direction (default: 0.0).
    Phi_theta0 : float, optional
        Initial phase in the theta direction (default: 0.0).
    Phi_r0 : float, optional
        Initial phase in the radial direction (default: 0.0).
    T : float, optional
        Total duration of the signal (default: 1.0).
    delta_T : float, optional
        Time step for the waveform (default: 10.0).
    eps : float, optional
        Accuracy parameter for the waveform generation (default: 1e-2).
    return_params : bool, optional
        Whether to return the parameters used for the signal generation (default: False).
    randomize : bool, optional
        Whether to randomize certain parameters (default: False).

    Returns
    -------
    signal : ndarray
        h plus and h cross of the gravitational wave signal.
    """

    if randomize:
        qS = np.arccos(np.random.uniform(-1, 1))
        phiS = np.random.uniform(0, 2 * np.pi)
        qK = np.arccos(np.random.uniform(-1, 1))
        phiK = np.random.uniform(0, 2 * np.pi)
        Phi_phi0 = np.random.uniform(0, 2 * np.pi)
        Phi_theta0 = np.random.uniform(0, 2 * np.pi)
        Phi_r0 = np.random.uniform(0, 2 * np.pi)

    m1, m2, a, Tpl, ef, x0 = params.T
    p_f = traj_gen.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, ef, 1.0) + 1e-6
    backwards_result = traj_gen(m1, m2, a, p_f, ef, x0, T=Tpl, integrate_backwards=True)
    p0_true = backwards_result[1][-1]
    e0_true = backwards_result[2][-1]
    x0_true = backwards_result[3][-1]

    if resp_gen:
        wave_gen_resp = get_response_generator(T, delta_T)
        print("Using response generator")
        wave_out = wave_gen_resp(m1, m2, a, p0_true, e0_true, x0_true, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0,
                            Phi_r0, T=T, dt=delta_T, padoutput=True, eps=eps)
        wave_out = wave_out[0] + 1j * wave_out[1]
    else:
        print("Using hp hx generator")
        wave_out = wave_gen(m1, m2, a, p0_true, e0_true, x0_true, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0,
                            Phi_r0, T=T, dt=delta_T, padoutput=True, eps=eps)

    if few.has_backend('gpu'):
        signal = wave_out.get()
        print("Waveform generated on GPU")
    else:
        signal = wave_out.copy()
        print("Waveform generated on CPU")

    time_vector = np.arange(signal.shape[0]) * delta_T
    params_dict = {'m1': m1, 'm2': m2, 'a': a, 'Tpl': Tpl, 'ef': ef, 'e0': e0_true, 'pf': p_f, 'p0': p0_true,
                   "dist": dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK, 'Phi_phi0': Phi_phi0,
                   'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0}
    if return_params:
        return time_vector, signal.real, signal.imag, params_dict
    else:
        return time_vector, signal.real, signal.imag


def generate_noise(signal_length, delta_T, psd):
    """
    Generate Gaussian noise in the time domain with a given PSD.

    Args:
        signal_length (int): Length of the time-domain signal.
        delta_T (float): Time step in seconds.
        psd (callable): Function that returns PSD values for given frequencies.

    Returns:
        np.ndarray: Time-domain noise realization.
    """
    frequencies = np.fft.rfftfreq(signal_length, delta_T)
    # Variance of the noise in the Fourier domain
    # (the extra 1/2 factor arises because we need to distribute the psd between the real and the imaginary parts of the signal)
    sigma2 = psd(frequencies) * signal_length * delta_T / 4
    n_tilde = (
        np.random.normal(0.0, sigma2**0.5, size=len(frequencies))
        + 1j * np.random.normal(0.0, sigma2**0.5, size=len(frequencies))
    )
    noise_real = np.fft.irfft(n_tilde, n=signal_length) / delta_T
    return noise_real


def generate_emri_signal_and_sfts(true_values, T_data, T_sft, delta_T, snr_ref, T_snr, psd, noise=True):
    """
    Generate EMRI signal, compute SFTs, and prepare data for analysis.

    Parameters:
    -----------
    true_values : np.ndarray
        EMRI parameters [m1, m2, a, Tpl, ef, x0]
    T_data : float
        Total data duration in years
    T_sft : float
        SFT duration in seconds
    delta_T : float
        Time step in seconds
    snr_ref : float
        SNR of the signal with duration T_snr
    T_snr : float
        Duration of the signal in years (it must always be bigger than or equal to T_data)

    Returns:
    --------
    dict : Dictionary containing:
        - 't': trimmed time array
        - 'signal_sfts': SFTs of the signal
        - 'noise_sfts': SFTs of noise realization
        - 'new_noise_sfts': SFTs of second noise realization
        - 'data_sfts': SFTs of signal + noise
        - 't_alpha': time samples for SFTs
        - 'samples_per_sft': number of samples per SFT
        - 'wind': window function
        - 'snr_final': final SNR of scaled signal
        - 'true_phi_f_fdot_fddot': frequency evolution parameters
        - 'm_hd' : Inner product <h|d> between signal SFTs and data SFTs
        - 'm_hh' : Inner product <h|h> (signal with itself)
        - 'm_hn' : Inner product <h|n> between signal SFTs and noise SFTs
    """
    # Generate EMRI signal
    t, hp, hx, param_dict = create_signal(true_values, T=T_snr, delta_T=delta_T, return_params=True, randomize=True, resp_gen=noise)
    snr_sig = get_snr(hp, delta_T, psd)  # Check SNR of the signal

    # Trim signal to desired duration
    mask = t < YRSID_SI * T_data
    hp, hx = snr_ref / snr_sig * hp[mask], snr_ref / snr_sig * hx[mask]  # Rescale signal to desired SNR
    t = t[mask]
    snr_final = get_snr(hp, delta_T, psd)
    print(f"Final SNR: {snr_final}")

    # Update distance
    param_dict['dist'] = param_dict['dist'] * snr_sig / snr_ref

    # SFT parameters
    samples_per_sft = int(T_sft / delta_T)
    wind = sp.signal.windows.tukey(samples_per_sft, 0.2)  # Window function

    # Compute SFTs
    signal_sfts = compute_sfts(hp, delta_T, wind, samples_per_sft)
    num_sfts = signal_sfts.shape[1]
    t_alpha = np.arange(num_sfts) * T_sft

    # Generate noise SFTs
    noise_td = generate_noise(hp.shape[0], delta_T, psd)
    if noise:
        data_td = hp + noise_td
    else:
        data_td = hp
    noise_sfts = compute_sfts(noise_td, delta_T, wind, samples_per_sft)
    new_noise_sfts = compute_sfts(generate_noise(hp.shape[0], delta_T, psd), delta_T, wind, samples_per_sft)
    if noise:
        data_sfts = signal_sfts + noise_sfts
    else:
        data_sfts = signal_sfts

    # Compute matched filtering
    freq = np.fft.rfftfreq(samples_per_sft, delta_T)
    m_hd = sft_inner_product(signal_sfts, data_sfts, freq, psd)
    m_hh = sft_inner_product(signal_sfts, signal_sfts, freq, psd)
    m_hn = sft_inner_product(signal_sfts, noise_sfts, freq, psd)

    # Get frequency evolution for true values
    true_phi_f_fdot_fddot = get_f_fdot_fddot_back(true_values, t_alpha)

    return {
        'data_td': data_td,
        'param_dict': param_dict,
        't': t,
        'data_sfts': data_sfts,
        'signal_sfts': signal_sfts,
        'noise_sfts': noise_sfts,
        'new_noise_sfts': new_noise_sfts,
        't_alpha': t_alpha,
        'samples_per_sft': samples_per_sft,
        'num_sfts': num_sfts,
        'wind': wind,
        'snr_final': snr_final,
        'true_phi_f_fdot_fddot': true_phi_f_fdot_fddot,
        'm_hd': m_hd,
        'm_hh': m_hh,
        'm_hn': m_hn
    }


def generate_injection_data(injection_params, T_DATA, T_SFT, DELTA_T, SNR_REF, T_SNR, psd, noise=True, seed=42):
    """
    Generate simulated EMRI signal and SFT data.

    Parameters
    ----------
    injection_params : dict
        Dictionary with keys: m1, m2, a, Tpl, ef
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - data_sfts: SFT data (signal + noise)
        - noise_sfts: Pure noise SFTs
        - signal_sfts: Pure signal SFTs
        - t_alpha: Time array
        - true_params: Array of true parameters [log10(m1), m2, a, Tpl, ef]
    """
    np.random.seed(seed)

    # Pack parameters for signal generation
    true_values = np.array([
        injection_params['m1'],
        injection_params['m2'],
        injection_params['a'],
        injection_params['Tpl'],
        injection_params['ef'],
        1.0  # x0 scaling factor
    ])

    # Generate signal and SFTs
    injection_dict = generate_emri_signal_and_sfts(
        true_values, T_DATA, T_SFT, DELTA_T, SNR_REF, T_SNR, psd, noise
    )

    # Extract relevant data
    t_alpha = injection_dict['t_alpha']
    data_sfts = np.asarray(injection_dict['data_sfts'], dtype=np.complex128)
    noise_sfts = np.asarray(injection_dict['noise_sfts'], dtype=np.complex128)
    signal_sfts = np.asarray(injection_dict['signal_sfts'], dtype=np.complex128)

    # True parameters in MCMC parameterization (log10(m1))
    true_params = np.array([
        np.log10(injection_params['m1']),
        injection_params['m2'],
        injection_params['a'],
        injection_params['Tpl'],
        injection_params['ef'],
    ])

    return {
        'data_sfts': data_sfts,
        'noise_sfts': noise_sfts,
        'signal_sfts': signal_sfts,
        't_alpha': t_alpha,
        'true_params': true_params,
        'injection_dict': injection_dict,
    }