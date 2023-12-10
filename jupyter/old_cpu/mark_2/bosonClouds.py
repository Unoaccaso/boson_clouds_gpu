import pandas as pd
from pandas import DataFrame

import numpy as np
from numpy import einsum
from numpy.random import default_rng

rng = default_rng()

from numexpr import evaluate

import xarray as xr

from scipy import signal

from functools import partial

import fast_histogram as fh


import matplotlib.pyplot as plt

# -------------------------- Constants
from astropy.constants import (
    G,
    hbar,
    M_sun,
    c,
    h,
)

G = G.value
hbar = hbar.value
c = c.value

# -------------- Custom modules
import sys

sys.path.append("./")
import distributionHelper as dh


def sim_signals(
    # CLOUDs PARAMETERS
    N_BH_per_clouds: int = 25000,
    clouds_center_kpc: np.ndarray = np.array([[8, 0]]),
    BH_distribution_in_clouds: str = "gauss",
    clouds_radius_kpc: float = 1e-3,
    # BH POPULATION PARAMETERS
    BH_mass_distribution: str = "kroupa",
    BH_mass_min: float = 1,
    BH_mass_max: float = 10,
    BH_age_distribution: str = "unif",
    BH_age_min_Yrs: float = float(1e4),
    BH_age_max_Yrs: float = float(1e6),
    spin_distribution: float = "unif",
    spin_min_value: float = 0.2,
    spin_max_value: float = 0.9,
    # BOSON GRID PARAMETERS
    N_bosons: int = 600,
    boson_grid_type: str = "log",
    boson_mass_min: float = -14,
    boson_mass_max: float = -12,
) -> DataFrame:
    # =========================================================================
    #                            INIZIALIZATION
    # =========================================================================
    clouds_center_kpc = np.array(clouds_center_kpc)
    N_clouds = clouds_center_kpc.shape[0]

    supported_BH_distributions_in_clouds = [
        "gauss",
        "unif",
        "fixed",
    ]
    supported_BH_mass_distributions = [
        "kroupa",
        "triang",
        "unif",
        "exp",
        "lin",
        "geom",
    ]
    supported_BH_age_distributions = [
        "unif",
        "gauss",
        "exp",
        "kroupa",
        "fixed",
        "logunif",
    ]
    supported_spin_distributions = [
        "unif",
        "gauss",
        "fixed",
    ]
    supported_boson_grid_types = [
        "log",
        "geom",
        "lin",
    ]

    if BH_distribution_in_clouds not in supported_BH_distributions_in_clouds:
        raise Exception(
            """
            Cannot populate the clouds with the desired distribution:\n
            Not supported yet!\n
            Supported distributions are: 
            """
            + f"{supported_BH_distributions_in_clouds}"
        )
    elif BH_mass_distribution not in supported_BH_mass_distributions:
        raise Exception(
            """
            Black Holes masses cannot be extracted with given distribution:\n
            Not supported yet!\n
            Supported distributions are:
            """
            + f"{supported_BH_mass_distributions}"
        )
    elif BH_age_distribution not in supported_BH_age_distributions:
        raise Exception(
            """
            Black Holes cannot be generated with given age distribution:\n
            Not supported yet!\n
            Supported distributions are: 
            """
            + f"{supported_BH_age_distributions}"
        )
    elif spin_distribution not in supported_spin_distributions:
        raise Exception(
            """
            Black Holes cannot be generated with given spin distribution:\n
            Not supported yet!\n
            Supported distributions are: 
            """
            + f"{supported_spin_distributions}"
        )

    elif boson_grid_type not in supported_boson_grid_types:
        raise Exception(
            """
            Cannot generate boson grid:\n
            Not supported!\n
            Supported grids are: 
            """
            + f"{supported_boson_grid_types}"
        )

    # =========================================================================
    #                              CONSTRUCTION
    # =========================================================================

    # --------------    Extracting BHs positions   --------------

    if BH_distribution_in_clouds == "gauss":  # -- GAUSSIAN POS DISTRIBUTION
        cov = [
            [clouds_radius_kpc / 100, 0],
            [0, clouds_radius_kpc / 100],
        ]
        positions = rng.multivariate_normal(
            mean=[0, 0],
            cov=cov,
            size=(N_clouds, N_BH_per_clouds),
        )
    elif BH_distribution_in_clouds == "unif":  # --- UNIFOM POS DISTRIBUTION
        phi = rng.uniform(
            low=0,
            high=2 * np.pi,
            size=(N_clouds, N_BH_per_clouds),
        )
        zero_to_one = rng.uniform(
            low=0,
            high=1,
            size=(N_clouds, N_BH_per_clouds),
        )
        rho = clouds_radius_kpc * np.sqrt(zero_to_one)
        positions = np.array([rho * np.cos(phi), rho * np.sin(phi)]).T
    elif BH_distribution_in_clouds == "fixed":  # ------ FIXED POS DISTRIBUTION
        positions = np.zeros((N_clouds, N_BH_per_clouds, 2))

    positions = (
        np.reshape(positions, (N_clouds, N_BH_per_clouds, 2))
        + clouds_center_kpc[:, np.newaxis, :]
    )

    # --------------  Calculating distances for each black hole  --------------

    xs = positions[:, :, 0]
    ys = positions[:, :, 1]
    ds = evaluate("sqrt(xs * xs + ys * ys)")

    # --------------------- Extracting Massess --------------------------------

    if BH_mass_distribution == "kroupa":
        masses = dh.kroupa(
            BH_mass_max,
            BH_mass_min,
            (N_clouds, N_BH_per_clouds),
        )
    elif BH_mass_distribution == "triang":
        masses = rng.triangular(
            BH_mass_min,
            (BH_mass_max - BH_mass_min) / 2,
            BH_mass_max,
            size=(N_clouds, N_BH_per_clouds),
        )
    elif BH_mass_distribution == "unif":
        masses = rng.uniform(
            low=BH_mass_min,
            high=BH_mass_max,
            size=(N_clouds, N_BH_per_clouds),
        )
    elif BH_mass_distribution == "exp":
        masses = dh.exponential(
            BH_mass_max,
            BH_mass_min,
            (N_clouds, N_BH_per_clouds),
        )
    elif BH_mass_distribution == "lin":
        masses = np.linspace(
            BH_mass_min,
            BH_mass_max,
            N_BH_per_clouds,
        ).reshape((N_clouds, N_BH_per_clouds))

    elif BH_mass_distribution == "geom":
        masses = np.geomspace(
            BH_mass_min,
            BH_mass_max,
            N_BH_per_clouds,
        ).reshape((N_clouds, N_BH_per_clouds))

    # --------------------- Extracting spins ----------------------------------

    if spin_distribution == "unif":
        spins = rng.uniform(
            low=spin_min_value,
            high=spin_max_value,
            size=(N_clouds, N_BH_per_clouds),
        )
    elif spin_distribution == "gauss":
        spins = dh.gaussian(
            spin_max_value,
            spin_min_value,
            (N_clouds, N_BH_per_clouds),
        )
    elif spin_distribution == "fixed":
        spins = np.full(
            (N_clouds, N_BH_per_clouds),
            fill_value=spin_min_value,
        )

    # --------------------- Extracting ages -----------------------------------

    if BH_age_distribution == "kroupa":
        BH_age = dh.kroupa(
            BH_age_max_Yrs,
            BH_age_min_Yrs,
            (N_clouds, N_BH_per_clouds),
        )
    elif BH_age_distribution == "triang":
        BH_age = rng.triangular(
            BH_age_min_Yrs,
            (BH_age_max_Yrs - BH_age_min_Yrs) / 2,
            BH_age_max_Yrs,
            size=(N_clouds, N_BH_per_clouds),
        )
    elif BH_age_distribution == "unif":
        BH_age = rng.uniform(
            low=BH_age_min_Yrs,
            high=BH_age_max_Yrs,
            size=(N_clouds, N_BH_per_clouds),
        )
    elif BH_age_distribution == "logunif":
        BH_age = rng.uniform(
            low=BH_age_min_Yrs,
            high=BH_age_max_Yrs,
            size=(N_clouds, N_BH_per_clouds),
        )
        BH_age = 10**BH_age
    elif BH_age_distribution == "exp":
        BH_age = dh.exponential(
            BH_age_max_Yrs,
            BH_age_min_Yrs,
            (N_clouds, N_BH_per_clouds),
        )
    elif BH_age_distribution == "gauss":
        BH_age = dh.gaussian(
            BH_age_max_Yrs,
            BH_age_min_Yrs,
            (N_clouds, N_BH_per_clouds),
        )
    elif BH_age_distribution == "fixed":
        BH_age = np.full(
            (N_clouds, N_BH_per_clouds),
            fill_value=BH_age_min_Yrs,
        )

    # --------------- Building Boson Grid -------------------------------------

    if boson_grid_type == "log":
        mus = np.logspace(
            start=boson_mass_min,
            stop=boson_mass_max,
            num=N_bosons,
        )
    elif boson_grid_type == "geom":
        mus = np.geomspace(
            start=boson_mass_min,
            stop=boson_mass_max,
            num=N_bosons,
        )
    elif boson_grid_type == "lin":
        mus = np.linspace(
            start=boson_mass_min,
            stop=boson_mass_max,
            num=N_bosons,
        )

    # =========================================================================
    #                             CALCULATIONS
    # =========================================================================

    Om0 = 2 * np.pi / 86400
    R0 = 5.5e6  # Rotational radius at Livingston (lower latitude)
    duty = 0.7  # Detectors duty cycle (approximate)
    Tobs = 365 * 86400 * duty  # Here we should use the exact fraction of non-zero data,
    onev = 1.60217653e-19

    const = G / (c**3 * hbar) * 2e30 * onev
    alpha = const * einsum("ij, k -> ikj", masses, mus).astype(np.float32)

    sq_alpha = einsum("ijk, ijk -> ijk", alpha, alpha).astype(np.float32)

    # elevation to 9th potency of alpha/0.1
    temp = alpha / 0.1  # pot == 0
    for i in range(3):
        temp = einsum("ijk, ijk -> ijk", temp, temp).astype(np.float32)
    # pot == 8
    pot_9 = einsum("ijk, ijk -> ijk", temp, alpha / 0.1).astype(np.float32)

    const = 27 * 86400 / 10.0
    tau_inst = const * einsum(
        "ijk, ik, ik -> ijk", 1 / pot_9, masses, 1 / spins
    ).astype(np.float32)

    # elevation to 15th potency of alpha/0.1
    temp = einsum("ijk, ijk -> ijk", temp, temp)  # pot == 1, dtype = np.float326
    pot_15 = einsum("ijk, ijk -> ijk", temp, 0.1 / alpha).astype(np.float32)

    const = 6.5e4 * 365 * 86400 / 10
    tau_gw = const * einsum("ijk, ik, ik -> ijk", 1 / pot_15, masses, 1 / spins).astype(
        np.float32
    )

    sq_masses = einsum("ik, ik -> ik", masses, masses).astype(np.float32)
    sq_mus = einsum("i, i -> i", mus, mus).astype(np.float32)
    freq = 483 * (
        1
        - 0.0056
        / (8 * 1e-22)
        * einsum("ij, k -> ikj", sq_masses, sq_mus).astype(np.float32)
    )
    freq = einsum("ijk -> ikj", freq).astype(np.float32)
    freq = einsum("ijk, k -> ijk", freq, mus / 1e-12).astype(np.float32)
    freq = einsum("ijk -> ikj", freq).astype(np.float32)

    const = c**3 / (2 * np.pi * G * 2e30)
    sq_spins = einsum("ij, ij -> ij", spins, spins).astype(np.float32)
    freq_max = const * spins
    _ = evaluate("1+sqrt(1 - sq_spins)")
    freq_max = einsum("ij, ij, ij -> ij", freq_max, masses, 1 / _).astype(np.float32)

    pot_17 = einsum("ijk, ijk -> ijk", temp, 0.1 / alpha).astype(np.float32)
    pot_17 = einsum("ijk -> ikj", pot_17).astype(np.float32)
    f_dot = 7e-15 * einsum("ijk, k -> ijk", pot_17, sq_mus).astype(np.float32)
    f_dot = einsum("ijk -> ikj", f_dot).astype(np.float32)

    freq = (
        freq
        + einsum("ijk, ik -> ijk", f_dot, BH_age * 365 * 86400).astype(np.float32)
        - einsum("ijk, ijk -> ijk", f_dot, tau_inst).astype(np.float32)
    )

    dfdot = Om0 * evaluate("2 * ceil(freq/10)") * 10 * R0 * c / (2 * Tobs / duty)

    den = evaluate("1/(1+4.0 * sq_alpha)")
    chi_c = 4 * einsum("ijk, ijk -> ijk", alpha, den).astype(np.float32)

    pot_7 = einsum("ijk, ijk, ijk -> ijk", pot_9, 0.1 / alpha, 0.1 / alpha).astype(
        np.float32
    )

    const = 1 / np.sqrt(3) * 3e-24 / (10 * 0.5)
    A = einsum("ijk, ik, ik -> ijk", pot_7, masses, spins).astype(np.float32)
    B = einsum("ijk, ik, ijk -> ijk", pot_7, masses, chi_c).astype(np.float32)
    amplitudes = evaluate("const * (A - B)")
    amplitudes = einsum("ijk, ik -> ijk", amplitudes, 1 / ds).astype(np.float32)
    timefactor = 1 + einsum("ij, ikj -> ikj", BH_age * 365 * 86400, 1 / tau_gw).astype(
        np.float32
    )
    timefactor = timefactor - einsum("ijk, ijk -> ijk", tau_inst, 1 / tau_gw).astype(
        np.float32
    )
    amplitudes = einsum("ijk, ijk -> ijk", amplitudes, 1 / timefactor).astype(
        np.float32
    )

    cond = np.array(
        (50 * tau_gw > BH_age[:, np.newaxis, :] * 365 * 86400)
        & (alpha < 0.1)
        & (freq > 20)
        & (freq < 2048)
        & (tau_inst < 10 * BH_age[:, np.newaxis, :] * 365 * 86400)
        & (10 * tau_inst < tau_gw)
        & (spins[:, np.newaxis, :] > chi_c)
        & (dfdot > f_dot)
    )
    # -------------------------------------------------------------------------

    clouds_dataSet = xr.Dataset(
        data_vars=dict(
            mass=(["cluster", "BH"], masses),
            # spin=(["cluster", "BH"], spins),
            # distance=(["cluster", "BH"], ds),
            age=(["cluster", "BH"], BH_age),
            frequency=(["cluster", "boson_mass", "BH"], freq * cond),
            frequency_uncut=(["cluster", "boson_mass", "BH"], freq),
            amplitude=(["cluster", "boson_mass", "BH"], amplitudes * cond),
            amplitude_uncut=(["cluster", "boson_mass", "BH"], amplitudes),
            alpha=(["cluster", "boson_mass", "BH"], alpha),
            tau_gw=(["cluster", "boson_mass", "BH"], tau_gw),
            tau_inst=(["cluster", "boson_mass", "BH"], tau_inst),
        ),
        coords=dict(
            boson_mass=(["boson_mass"], mus),
        ),
    )
    return clouds_dataSet.where(clouds_dataSet != 0)


def inject_signals_and_CR(
    freqs,
    amps,
    Tfft,
    cluster_idx=0,
    band_size=10,
    noise_amp=1e-25,  # seconds  # Hz
):
    # Extracting frequency bins
    df = 1 / Tfft

    left_border_freq = np.floor(np.nanmin(freqs, axis=1) / band_size) * band_size
    right_border_freq = left_border_freq + band_size
    nsteps = np.ceil(band_size / df)
    bins = np.linspace(left_border_freq, right_border_freq, int(nsteps) + 1).T
    hist_by_row = list(
        zip(
            *map(
                np.histogram,
                freqs,
                bins,
                [None] * freqs.shape[0],
                [None] * freqs.shape[0],
                amps,
            )
        )
    )

    hists = np.array(hist_by_row[0])
    theres_signals = hists > 0
    bins = np.array(hist_by_row[1])

    if isinstance(noise_amp, float):
        hists_with_noise = hists + rng.exponential(noise_amp, hists.shape)
    else:
        mid_freqs = (
            np.cumsum(
                np.diff(bins, axis=1),
                axis=1,
            )
            / 2
            + bins[:, 0, None]
        )
        freqs_of_noise = closest_argmin(bins, noise_amp[:, 0])
        noise_amplitude = noise_amp[freqs_of_noise, 1]
        noise = rng.exponential(noise_amplitude[:, :-1], hists.shape)
        hists_with_noise = hists + noise

    mean = np.nanmean(hists_with_noise, axis=1)
    sigma = np.nanstd(hists_with_noise, axis=1)
    all_CRs = (hists_with_noise - mean[:, None]) / sigma[:, None]
    signal_CRs = np.where(theres_signals, all_CRs, np.NaN)

    """
    mean = np.nanmean(noise, axis=1)
    sigma = np.nanstd(noise, axis=1)
    all_CRs = (noise - mean[:, None]) / sigma[:, None]
    signal_CRs = np.where(theres_signals, all_CRs, np.NaN)
    """

    return hists_with_noise, bins, all_CRs, signal_CRs


def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx == L] = L - 1
    mask = (sorted_idx > 0) & (
        (np.abs(A - sorted_B[sorted_idx - 1]) < np.abs(A - sorted_B[sorted_idx]))
    )
    return sidx_B[sorted_idx - mask]
