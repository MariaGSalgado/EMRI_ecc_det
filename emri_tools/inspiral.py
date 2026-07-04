import numpy as np
import scipy as sp
from collections import defaultdict

from few.utils.geodesic import get_separatrix
from few.utils.constants import MRSUN_SI


def compute_trajectory(m1, m2, a, Tpl, e_f, xI, traj_gen):
    """Computes the inspiral trajectory for an EMRI."""

    # Determine final semi-latus rectum
    p_f = traj_gen.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, e_f, xI) + 1e-6

    # Compute trajectory
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj_gen(m1, m2, a, p_f, e_f, xI, T=Tpl, integrate_backwards=True)

    return {
        "t": t,
        "p": p,
        "e": e,
        "xI": x,
        "Phi_phi": Phi_phi,
        "Phi_theta": Phi_theta,
        "Phi_r": Phi_r
    }


def compute_modes(a, p, e, xI, theta, phi, amp_gen, ylm_gen, specific_modes=None):
    """Computes the Teukolsky amplitudes and corresponding spherical harmonics for each (l, m, n) mode."""

    # Get amplitudes along trajectory
    teuk_modes = amp_gen(a, p, e, xI, specific_modes=specific_modes)

    # Get ylms (spherical harmonics)
    ylms = ylm_gen(amp_gen.unique_l, amp_gen.unique_m, theta, phi).copy()[amp_gen.inverse_lm]

    return teuk_modes, ylms


def mode_energy_inspiral(m1, m2, a, Tpl, e_f, xI, theta_K, phi_K, traj_gen, amp_gen, ylm_gen):
    """Computes the energy of each (m,n) mode summed over the full inspiral, normalized by the sum over all the modes."""

    # Compute trajectory of the smaller body: p(t), e(t), xI(t), Phi_phi(t), Phi_theta(t), Phi_r(t)
    out = compute_trajectory(m1, m2, a, Tpl, e_f, xI, traj_gen)
    t, p, e, xI = out["t"], out["p"], out["e"], out["xI"]
    dt = np.diff(t, append=t[-1])

    # Get amplitudes along trajectory and spherical harmonics
    teuk, ylms = compute_modes(a, p, e, xI, theta_K, phi_K, amp_gen, ylm_gen)
    n_modes = teuk.shape[1]
    ylms_pos = ylms[:n_modes]
    ylms_neg = ylms[n_modes:]

    m0mask = amp_gen.m0mask
    nonzero_indices = np.where(m0mask)[0]

    # Find the indices of the modes with the same (m,n)
    mn_indices = defaultdict(list)
    for idx, (l, m, n) in enumerate(zip(
            amp_gen.l_arr_no_mask,
            amp_gen.m_arr_no_mask,
            amp_gen.n_arr_no_mask)):
        mn_indices[(m, n)].append(idx)

    # Obtain the energy of each (m,n) mode during the inspiral (not scaled)
    mn_list = []
    mode_energy_insp = []

    for (m, n), indices in mn_indices.items():

        indices = np.array(indices)

        H_pos = np.sum(teuk[:, indices] * ylms_pos[indices], axis=1)

        if m != 0:
            ylms_neg_indices = np.searchsorted(nonzero_indices, indices)
            l_arr = amp_gen.l_arr_no_mask[indices]
            parity = (-1.0) ** l_arr
            H_neg = np.sum(parity * np.conj(teuk[:, indices]) * ylms_neg[ylms_neg_indices], axis=1)
        else:
            H_neg = 0

        power = np.abs(H_pos) ** 2 + np.abs(H_neg) ** 2
        energy_insp = np.sum(power * dt)
        mode_energy_insp.append(energy_insp)
        mn_list.append((m, n))

    mode_energy_insp = np.array(mode_energy_insp)
    mn_list = np.array(mn_list)

    # Compute relative contributions
    rel_mode_energy_insp = mode_energy_insp / np.sum(mode_energy_insp)

    return mn_list, rel_mode_energy_insp


def mode_snr2_inspiral(m1, m2, a, Tpl, e_f, xI, dist, theta_K, phi_K, psd, traj_gen, amp_gen, ylm_gen):
    """Computes the SNR^2 of each (m,n) mode along the inspiral."""

    # Compute trajectory of the smaller body: p(t), e(t), xI(t), Phi_phi(t), Phi_theta(t), Phi_r(t)
    out = compute_trajectory(m1, m2, a, Tpl, e_f, xI, traj_gen)
    t, p, e, xI = out["t"], out["p"], out["e"], out["xI"]
    dt = np.diff(t, append=t[-1])

    # Obtain the fundamental frequencies during the inspiral
    freqs = traj_gen.inspiral_generator.eval_integrator_derivative_spline(t, order=1)[:, 3:6] / (2 * np.pi)
    f_phi, f_theta, f_r = freqs.T

    # Get amplitudes along trajectory and spherical harmonics
    teuk, ylms = compute_modes(a, p, e, xI, theta_K, phi_K, amp_gen, ylm_gen)
    n_modes = teuk.shape[1]
    ylms_pos = ylms[:n_modes]
    ylms_neg = ylms[n_modes:]

    m0mask = amp_gen.m0mask
    nonzero_indices = np.where(m0mask)[0]

    # Find the indices of the modes with the same (m,n)
    mn_indices = defaultdict(list)
    for idx, (l, m, n) in enumerate(zip(
            amp_gen.l_arr_no_mask,
            amp_gen.m_arr_no_mask,
            amp_gen.n_arr_no_mask)):
        mn_indices[(m, n)].append(idx)

    # Compute the factor to convert from Gpc to solar masses (geometric units)
    mu = (m1 * m2) / (m1 + m2)
    d_L = dist * sp.constants.parsec * 1e9 / MRSUN_SI
    scale = mu / d_L

    # Obtain the SNR^2 of each (m,n) mode during the inspiral
    mn_list = []
    mode_snr2 = []

    for (m, n), indices in mn_indices.items():

        indices = np.array(indices)

        H_pos = np.sum(teuk[:, indices] * ylms_pos[indices], axis=1)

        if m != 0:
            ylms_neg_indices = np.searchsorted(nonzero_indices, indices)
            l_arr = amp_gen.l_arr_no_mask[indices]
            parity = (-1.0) ** l_arr
            H_neg = np.sum(parity * np.conj(teuk[:,indices]) * ylms_neg[ylms_neg_indices], axis=1)
        else:
            H_neg = 0

        power = np.abs(H_pos)**2 + np.abs(H_neg)**2
        mode_freqs = np.abs(m*f_phi + n*f_r)
        psd_vals = psd(mode_freqs)
        snr2 = 4 * scale ** 2 * np.sum((power / psd_vals) * dt)

        mn_list.append((m, n))
        mode_snr2.append(snr2)

    return np.array(mn_list), np.array(mode_snr2)


def find_total_snr(m1, m2, a, Tpl, e_f, xI, dist, theta_K, phi_K, psd, traj_gen, amp_gen, ylm_gen):
    """Computes the total SNR for a given inspiral."""

    mn_list, mode_snr2 = mode_snr2_inspiral(m1, m2, a, Tpl, e_f, xI, dist, theta_K, phi_K, psd, traj_gen, amp_gen, ylm_gen)
    total_snr = np.sqrt(mode_snr2.sum())
    return total_snr


def find_relevant_modes(m1, m2, a, Tpl, e_f, xI, dist, theta_K, phi_K, psd, threshold_snr, traj_gen, amp_gen, ylm_gen):
    """Finds the modes that make up a certain threshold on the total SNR
    and computes the respective weights, individual and joint SNR."""

    mn_list, mode_snr2 = mode_snr2_inspiral(m1, m2, a, Tpl, e_f, xI, dist, theta_K, phi_K, psd, traj_gen, amp_gen, ylm_gen)

    # Relative contributions
    weights = mode_snr2 / np.sum(mode_snr2)

    # Sort modes by importance
    sorted_indices = np.argsort(-weights)
    sorted_modes = mn_list[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_snr2 = mode_snr2[sorted_indices]

    # Select modes above threshold SNR
    threshold_snr2 = threshold_snr ** 2
    cum = np.cumsum(sorted_weights)
    idx = np.argmax(cum >= threshold_snr2)

    selected_modes = sorted_modes[:idx + 1]
    selected_weights = sorted_weights[:idx + 1]
    selected_snr = np.sqrt(sorted_snr2[:idx + 1])
    total_snr = np.sqrt(mode_snr2.sum())

    return selected_modes, selected_weights, selected_snr, total_snr