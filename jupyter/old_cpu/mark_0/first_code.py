# bhogen.py
import numpy as np
from numpy import random as rnd
from time import time
from time import sleep
from termcolor import colored
import psutil
import os
import matplotlib.pyplot as plt


process = psutil.Process(os.getpid())


class bosonGrid:
    # This class generates a boson grid mass
    def __init__(
        self,
        n_mus=300,
        mu_min=2.15e-13,
        mu_max=1.27e-12,
        scale="lin",  # can be log or lin (for linearly or logarithmically spaced points)
    ):
        self.n_mus = n_mus
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.scale = scale
        self.boson_grid = None
        self.mass_grid_built = False

    def build_mass_grid(self):
        # print("Building mass grid...")
        if self.scale == "log":
            self.boson_grid = np.geomspace(self.mu_min, self.mu_max, self.n_mus)
            # print("=> Mass Grid was built, a logarithmic scale was used")

        elif self.scale == "lin":
            self.boson_grid = np.linspace(self.mu_min, self.mu_max, self.n_mus)
            # print("=> Mass Grid was built, a linear scale was used")
        self.mass_grid_built = True


# ==============================================================================
# ==============================================================================
class bhogen(bosonGrid):
    def __init__(
        self,
        obs_distance=200,  # distance [kpc] from detector
        cluster_eta=1e3,  # system age [yr]
        Nbh=20000,  # Number of generated BH systems
        Nbins=36,
        mass_dis="kroupa",  # BH mass distribution (currently, uniform ('unif'), exponential ('exp'), Kroupa ('kroupa') and Triangular ('triang') are supported)
        mass_range=[5, 25],
        spin_dis="gauss",  # BH spin distribution (currently unifomr ('unif), gaussian ('gauss') are supported)
        spin_range=[0.2, 0.9],  # in case of uniform distr
        spin_mean=0.4,  # Mean and sigma are used for gaussian distr
        spin_sigma=0.2,
        multiple_runs=True,  # one has the possibility of performing multiple runs
        Nrun=10,
        # Boson Grid Parameters
        n_mus=300,
        mu_min=2.15e-13,
        mu_max=1.27e-12,
        scale="lin",  # can be log or lin (for uniformly or logarithmically spaced points)
    ):
        # Check for input errors
        if spin_dis not in ["unif", "gauss"]:
            raise Exception(
                "The spin distribution must be uniform ('unif') or Gaussian ('gauss')"
            )
        if mass_dis not in ["unif", "exp", "kroupa", "triang"]:
            raise Exception(
                "Supported distributions for BH masses are: \nuniform ('unif')\nexponential ('exp')\nkroupa ('kroupa')\ntriang('triang')"
            )
        if scale not in ["lin", "log"]:
            raise Exception(
                "The points on the boson mass grid can be distributed logaritmically ('log') or uniformly ('lin')"
            )
        # ==================== INITIALIZATION START ============================

        # Cosmological Parameters
        self.obs_distance = obs_distance
        self.cluster_eta = cluster_eta
        self.cluster_eta_sec = cluster_eta * 365 * 86400

        self.Nbh = Nbh

        # Multiple runs
        self.Nbins = Nbins
        self.Nrun = Nrun

        # BH Masses
        self.mass_dis = mass_dis
        self.Mbh_min = mass_range[0]
        self.Mbh_max = mass_range[1]
        # BH Spins
        self.spin_dis = spin_dis
        self.Sbh_min = spin_range[0]
        self.Sbh_max = spin_range[1]
        self.Sbh_mean = spin_mean
        self.Sbh_sigma = spin_sigma

        self.Bhs = None

        self.saved_freqs = None
        self.saved_amps = None

        self.saved_hists_counts = None
        self.saved_hists_bins = None
        self.saved_freqs_variance = None
        self.saved_top_bins = None
        self.saved_rsigma_freq = None
        self.saved_lsigma_freq = None

        """Here I instantiate a boson mass grid inside the cluster class, 
        so that i have a list of masses inside this class"""
        bosonGrid.__init__(self, n_mus=n_mus, mu_min=mu_min, mu_max=mu_max, scale=scale)
        """ ATTENTION!!!
        Here is building the grid, if this is not done correctly no error 
        will occur, the mass grid inside the cluster will be an array of zeros"""

        self.cluster_populated = False
        self.wave_emitted = False
        self.freq_distr_calculated = False

    # ======================= INITIALIZATION END ===============================
    # Defining constants

    G = 6.67e-11
    c = 299792458
    Om0 = 2 * np.pi / 86400  # 1/day
    R0 = 5.5e6  # Rotational radius at Livingston (lower latitude)
    hbar = 1.054571e-34
    onev = 1.60217653e-19
    fint = 1e30  # Small interaction regime.

    duty = 0.681  # Detectors duty cycle (approximate)
    Tobs = 365 * 86400 * duty  # Here we should use the exact fraction of non-zero data,

    def populate(self, export_masses=False):
        """Populating the BH array with randomly extracted masses"""
        if self.mass_dis == "unif":
            masses = rnd.uniform  # (self.Mbh_min, self.Mbh_max, self.Nbh)
            Mbh_ave = np.mean(self.Bhs[0])
            # print(#Black Holes born correctly")
            # print(## #=> You now own a population of {self.Nbh} Black holes")
            # print#(
            # #=> Masses are uniformly distributed from {self.Mbh_min} to {self.Mbh_max} solar masses"
            # )

        elif self.mass_dis == "exp":
            Mbh_ave = self.Mbh_max - self.Mbh_min
            R1 = 1 - np.exp(-self.Mbh_min / Mbh_ave)
            R2 = 1 - np.exp(-self.Mbh_max / Mbh_ave)
            R = rnd.uniform(R1, R2, self.Nbh)
            masses = -Mbh_ave * np.log(1 - R)
            # print(#Black Holes born correctly")
            # print(## #=> You now own a population of {self.Nbh} Black holes")
            # print#(
            # #=> Masses are exponentially distributed from {self.Mbh_min} to {self.Mbh_max} solar masses"
            ##)

        elif self.mass_dis == "kroupa":
            a = 2.3
            Mbh_unif = rnd.uniform(0, 1, self.Nbh)
            K = (1 - a) / (self.Mbh_max ** (1 - a) - self.Mbh_min ** (1 - a))
            Y = ((1 - a) / K * Mbh_unif + self.Mbh_min ** (1 - a)) ** (1 / (1 - a))
            jj = [(Y > self.Mbh_min) & (Y < self.Mbh_max)]
            masses = Y[tuple(jj)]
            # print(#Black Holes born correctly")
            # print(## #=> You now own a population of {self.Nbh} Black holes")
            # print#(
            # #=> Masses are distributed with kroupa method from {self.Mbh_min} to {self.Mbh_max} solar masses"
            ##)

        elif self.mass_dis == "triang":
            masses = rnd.triangular(
                self.Mbh_min, self.Mbh_max - self.Mbh_min, self.Mbh_max, self.Nbh
            )
            Mbh_ave = self.Mbh_max - self.Mbh_min
            # print(#Black Holes born correctly")
            # print(## #=> You now own a population of {self.Nbh} Black holes")
            # print#(
            # #=> Masses are triangularly distributed from {self.Mbh_min} to {self.Mbh_max} solar masses"
            ##)

        # Populating the BH array with randomly extracted spins

        if self.spin_dis == "unif":
            spins = rnd.uniform(self.Sbh_min, self.Sbh_max, self.Nbh)
            # print(
            # #=> Your Black Holes now have random spin uniformly distributed from {self.Sbh_min} to {self.Sbh_max}.\n"
            ##)
        elif self.spin_dis == "gauss":
            """
            Attention, by simply constructing a np array extracting from a gaussian
            distribution, it is possible to extract values of spin out of given range,
            instead we prebuild an array and randomly extract from that
            """
            step = (self.Sbh_max - self.Sbh_min) / self.Nbh
            _ = np.arange(self.Sbh_min, self.Sbh_max, step)
            gaussian = rnd.normal(self.Sbh_mean, self.Sbh_sigma, int(1e6))
            h, bin_edges = np.histogram(gaussian, bins=self.Nbh, density=True)
            p = h * np.diff(bin_edges)
            spins = rnd.choice(_, size=self.Nbh, p=p)
            # print(
            # #=> Your Black Holes now have random spin with mean value {self.Sbh_mean}, a Gaussian distribution was used.\n"
            # )

        self.Bhs = np.array([masses, spins])
        self.cluster_populated = True

        if export_masses:
            # Returns an array of black holes masses and spin: k-th BH is Bhs[k][mass, spin]
            return self.Bhs

    def emit_GW(self, remove_undetectable=True):
        if self.cluster_populated & self.mass_grid_built:
            # print("\nEmission of Gravitational Waves...")

            Mbhs = self.Bhs[0, :]
            Sbhs = self.Bhs[1, :]
            mus = self.boson_grid

            alpha = (
                self.G
                / (self.c**3 * self.hbar)
                * 2e30
                * Mbhs
                * mus[:, np.newaxis]
                * self.onev
            )
            chi_c = 4 * alpha / (1 + 4.0 * alpha**2)

            checkpoint1 = time()

            temp0 = 1 / ((alpha / 0.1) * (alpha / 0.1))
            temp1 = temp0 * temp0 * temp0 * temp0 / (alpha / 0.1)

            temp2 = temp1 * temp0 * temp0 * temp0

            tau_inst = 27 * 86400 / 10.0 * Mbhs * temp1 / Sbhs
            tau_gw = 3 * 6.5e4 * 365 * 86400 * Mbhs / 10 * temp2 / Sbhs

            freq = (
                483
                * (mus[:, np.newaxis] / 1e-12)
                * (
                    1
                    - 0.0056
                    / 8
                    * (Mbhs / 10.0) ** 2
                    * (mus[:, np.newaxis] / 1e-12) ** 2
                )
            )
            newtime = time()
            # print(
            # #=> {freq.shape[0] * freq.shape[1]:.0E} Frequencies calculated - t: {newtime - newtime:.2f} s"
            ##)

            freq_max = (
                self.c**3.0
                / (2 * np.pi * self.G * 2e30 * Mbhs)
                * Sbhs
                / (1 + np.sqrt(1 - Sbhs**2))
            )  # ------ Maximum allowed GW frequency

            # print(
            # 1 fino a qui tutto bene: {process.memory_info().rss / (1024*1024*1024)}"
            ##)

            # roba 1
            fdot = (
                7e-15 * (alpha[:] / 0.1) ** (17) * (mus[:, np.newaxis] / 1e-12) ** 2
                + 1e-10
                * (10**17 / self.fint) ** 4
                * (alpha / 0.1) ** (17)
                * (mus[:, np.newaxis] / 1e-12) ** 2
            )  # roba 2

            # print(
            # 2 fino a qui tutto bene: {process.memory_info().rss / (1024*1024*1024)}"
            ##)

            # print(
            # 3 fino a qui tutto bene: {process.memory_info().rss / (1024*1024*1024)}"
            # )

            freq_now = freq + fdot * (self.cluster_eta_sec - tau_inst)
            # print(
            # 4 fino a qui tutto bene: {process.memory_info().rss / (1024*1024*1024)}"
            # )

            # print(
            # 5 fino a qui tutto bene: {process.memory_info().rss / (1024*1024*1024)}"
            # )

            dfr = self.Om0 * np.sqrt(
                2 * np.ceil(freq_now / 10) * 10 * self.R0 / self.c
            )  # -------------------------------------------- Search frequency bin

            # print(
            # 6 fino a qui tutto bene: {process.memory_info().rss / (1024*1024*1024)}"
            # )
            dfdot = dfr / (2 * self.Tobs / self.duty)

            # print("\nCalculating wave amplitudes...")
            t = time()
            h0 = (
                1
                / np.sqrt(3)
                * 3.0e-24
                / 10
                * Mbhs
                * (alpha[:] / 0.1) ** 7
                * (Sbhs - chi_c)
                / 0.5
            )  # --- GW peak amplitude at d=1 kpc
            h0 = h0 / self.obs_distance

            timefactor = (
                1 + (self.cluster_eta_sec - tau_inst) / tau_gw
            )  # --------------------------- Time-dependent reduction factor
            h0 = h0 / timefactor

            """
            conditions to be met in order to have a potentially detectable signal
            (there may be some redundance)

            o tau_inst < t0s          : superradiance time scale must be shorter than system age
            o freq < freq_max         : condition for the development of the instability
            o 10*tau_inst < tau_gw    : we want the instability is fully completed
            o chi_i > chi_c           : condition for the development of the instability
            o (freq>20) & (freq<610)  : GW frequency in the search band
            o dfdot > fdot            : signal spin-up within half bin
            """
            self.wave_emitted = True

            # print(# #=> Gravitational Waves emitted, elapsed time: {time() - t:.2f} s")#

            if remove_undetectable:
                # print("\nSelecting detectable Waves...")
                t = time()
                cond = np.array(
                    (tau_inst < self.cluster_eta_sec)
                    & (freq > 20)
                    & (freq < 610)
                    & (freq < freq_max)
                    & (10 * tau_inst < tau_gw)
                    & (Sbhs > chi_c)
                    & (dfdot > fdot)
                    & (freq_now < 610)
                )

                # Applying conditions
                Mbhs = Mbhs * cond[:]  # This is now a matrix
                Sbhs = Sbhs * cond[:]  # This is now a matrix
                freq_now = freq_now * cond
                h0 = h0 * cond

                # Removing boson mass that didn't produce any wave
                parser = np.any(Mbhs, axis=1)
                Mbhs = Mbhs[parser]
                Sbhs = Sbhs[parser]
                freq_now = freq_now[parser]
                h0 = h0[parser]

                # print(
                # #=> {self.n_mus - Mbhs.shape[0]} points were removed from the grid"
                # )
                self.boson_grid = self.boson_grid[parser]
                # print(# #=> Grid Updated - elapsed time: {time() - t:.2f} s")#

            # Updating stored data
            # print("\nSaving data ...")
            """
            The code used leaves a 0 in places of BHs that didn't produce observable
            waves, by masking the arrays those values will not be considered in calculations.
            It is the fastest way to remove those data.
            """
            self.Bhs = np.array([Mbhs, Sbhs])
            self.saved_freqs = freq_now
            self.saved_amps = h0
            # print("=> Data saved\n\n")

    # Some functions to extract data from the cluster
    def get_masses(self):
        if self.cluster_populated:
            # returns a 2D array, every row is the array of masses that produced
            # detectable waves
            return np.ma.masked_equal(self.Bhs[0], 0)
        else:
            raise Exception("Cluster was not populated, run cl.populate() before")

    def get_spins(self):
        if self.cluster_populated:
            return np.ma.masked_equal(self.Bhs[1], 0)
        else:
            raise Exception("Cluster was not populated, run cl.populate() before")

    def get_freqs(self):
        if self.wave_emitted:
            return np.ma.masked_equal(self.saved_freqs, 0)
        else:
            raise Exception("Cluster was not populated, run cl.emit_GW() before")

    def get_amplitudes(self):
        if self.wave_emitted:
            return np.ma.masked_equal(self.saved_amps, 0)
        else:
            raise Exception("Cluster was not populated, run cl.emit_GW() before")

    def get_freq_variance(self):
        if self.freq_distr_calculated:
            return self.saved_freqs_variance
        else:
            raise Exception(
                "Frequency distribution was not calculated, run cl.calc_freq_distr() before."
            )

    # ==========================================================================
    """
    Make a function to count len by automatically skip the masked values
    """
    # ==========================================================================

    def plot_freq_distr(self, mu, yscale="log", show_sigma=True):
        if self.freq_distr_calculated:
            if (mu < self.mu_max) & (mu > self.mu_min):
                mu = np.argmin(np.abs(self.boson_grid - mu))
                mu_value = self.boson_grid[mu]
                bins = self.saved_hists_bins[mu][:-1]
                counts = self.saved_hists_counts[mu]
                bin_size = np.diff(self.saved_hists_bins[mu])

                top_bin = self.saved_top_bins[mu]
                lsigma = self.saved_lsigma_freq[mu]
                rsigma = self.saved_rsigma_freq[mu]

                plt.bar(bins, counts, width=bin_size, align="edge")
                plt.yscale(yscale)

                plt.title(
                    # Histogram of frequencies for boson mass of ${mu_value*1e12:.2f}$"
                    # + "$\cdot 10^{-12} eV$"
                )
                plt.xlabel("Frequency $[Hz]$")
                plt.ylabel("occ/prob")

                if show_sigma:
                    plt.vlines(top_bin, 0, counts.max())
                    plt.vlines(rsigma, 0, counts.max())
                    plt.vlines(lsigma.max(), 0, counts.max())
                plt.show()
            else:
                raise Exception("Please insert a reasonable value for boson mass")
        else:
            raise Exception(
                "Frequency distribution was not calculated, run cl.calc_freq_distr() before"
            )
