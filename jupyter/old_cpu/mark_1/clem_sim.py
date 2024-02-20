# clem_sim.py

"""
CLUSTER EMISSION SIMULATION
This package can be used to simulate the GW emission from ultralight boson clouds,
formed around black holes, in a dense globular clusters.

Choosing an array of masses for the boson (the so called 'Boson Grid'), it is possible
to perform the simulation of the emission and the recostruction of the spectrum in histograms.

R. Felicetti (2022)
"""

# Importing the necessary libraries

import trace
import numpy as np
from numpy import random as rnd
from time import time
from termcolor import colored

import matplotlib.pyplot as plt


class bosonGrid:
    # Used to generate the boson grid.
    #
    # Note that when instantiating a grid, nothing is populated during initialization,
    # so remember to call the building method.

    def __init__(
        self,
        n_mus=300,
        mu_min=2.15e-13,
        mu_max=1.27e-12,
        scale="lin",  # can be log or lin (for linearly or logarithmically spaced points)
        verbose=False,
        v=False,
    ):
        """
        Input values:
            n_mus: is the number of masses to inspect (the length of the array)
            mu_min: minimum mass value
            mu_max: maximum mass value
            scale: one can choose between log or linearly distributed mass
        """
        self.n_mus = n_mus
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.scale = scale
        self.boson_grid = None
        self.mass_grid_built = False  # This updates if the grid gets populated
        self.verbose = verbose
        self.v = v

    def build_mass_grid(self):
        # This method populates the grid and stores the values in a numpy array
        # Also the state of the grid gets updated to built.

        if self.verbose or self.v:
            print("Building mass grid...")

        if self.scale == "log":
            self.boson_grid = np.logspace(self.mu_min, self.mu_max, self.n_mus)
            if self.verbose or self.v:
                print("=> Mass Grid was built, a logarithmic scale was used")

        elif self.scale == "lin":
            self.boson_grid = np.linspace(self.mu_min, self.mu_max, self.n_mus)
            if self.verbose or self.v:
                print("=> Mass Grid was built, a linear scale was used")

        self.mass_grid_built = True


# ==============================================================================
# ==============================================================================
class cluster(bosonGrid):
    # This is the main class of the package.
    # Used to generate and populate the cluster, emit GWs and generate the desired histograms
    # An example of usage can be found in README.TXT

    def __init__(
        self,
        obs_distance=4,  # distance [kpc] from detector
        cluster_eta=1e10,  # system age [yr]
        Nbh=10000,  # Number of generated BH systems
        Nbins=36,
        mass_dis="kroupa",  # BH mass distribution (currently, uniform ('unif'), exponential ('exp'), Kroupa ('kroupa') and Triangular ('triang') are supported)
        mass_range=[5, 25],
        spin_dis="gauss",  # BH spin distribution (currently unifomr ('unif), gaussian ('gauss') are supported)
        spin_range=[0.2, 0.9],  # in case of uniform distr
        # Mean and sigma are used for gaussian distr
        spin_mean=0.4,
        spin_sigma=0.2,
        # Boson Grid Parameters
        n_mus=300,
        mu_min=2.15e-13,
        mu_max=1.27e-12,
        scale="lin",  # can be log or lin (for uniformly or logarithmically spaced points)
        verbose=False,
        v=False,
    ):
        # Check for input errors
        if spin_dis not in ["unif", "gauss", "fixed"]:
            raise Exception(
                "The spin distribution must be uniform ('unif') or Gaussian ('gauss')"
            )
        if mass_dis not in ["unif", "exp", "kroupa", "triang", "array"]:
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
        self.cluster_eta_sec = (
            cluster_eta * 365 * 86400
        )  # Converting the age of the cluster in seconds

        self.Nbh = Nbh  # Number of black holes in the cluster

        # Multiple runs
        self.Nbins = Nbins

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

        # Characteristic times ref. to theory for those
        self.tau_inst = 0
        self.tau_gw = 0

        self.Bhs = None  # This will contain the arrays of mass and spin of the BHs

        # Unmasked mass array (May be useful)
        self.massesUM = 0

        self.saved_freqs = None
        self.saved_amps = None

        self.saved_hists_counts = None
        self.saved_hists_bins = None
        self.saved_freqs_variance = None
        self.saved_top_bins = None
        self.saved_rsigma_freq = None
        self.saved_lsigma_freq = None
        self.chi_c = None
        self.alpha = None

        # Here I instantiate a boson mass grid inside the cluster class,
        # so that i have a list of masses bounded to the cluster
        bosonGrid.__init__(
            self,
            n_mus=n_mus,
            mu_min=mu_min,
            mu_max=mu_max,
            scale=scale,
            verbose=verbose,
            v=v,
        )

        """ ATTENTION!!!
        Here is building the grid, if this is not done correctly no error 
        will occur, the mass grid inside the cluster will be an array of zeros"""

        self.cluster_populated = False
        self.wave_emitted = False
        self.freq_distr_calculated = False

    # ======================= INITIALIZATION END ===============================

    # Defining physical constants

    G = 6.67e-11
    c = 299792458
    Om0 = 2 * np.pi / 86400  # 1/day
    R0 = 5.5e6  # Rotational radius at Livingston (lower latitude)
    hbar = 1.054571e-34
    onev = 1.60217653e-19
    fint = 1e30  # Small interaction regime.

    duty = 0.7  # Detectors duty cycle (approximate)
    Tobs = 365 * 86400 * duty  # Here we should use the exact fraction of non-zero data,

    def populate(self, export_BH_data=False, verbose=False, v=False):
        # This method populates the cluster with a random selection of BHs.
        # It randomly extracts the masses in the giving range and couples them
        # to a random value of the spin.

        """Populating the BH array with randomly extracted masses"""
        if self.mass_dis == "unif":
            masses = rnd.uniform(self.Mbh_min, self.Mbh_max, self.Nbh)
            Mbh_ave = np.mean(self.Bhs[0])
            if verbose or v:
                print(f"Black Holes born correctly")
                print(f"=> You now own a population of {self.Nbh} Black holes")
                print(
                    f"=> Masses are uniformly distributed from {self.Mbh_min} to {self.Mbh_max} solar masses"
                )

        elif self.mass_dis == "array":
            masses = np.linspace(self.Mbh_min, self.Mbh_max, self.Nbh)
            if verbose or v:
                print(f"Black Holes born correctly")
                print(f"=> You now own a population of {self.Nbh} Black holes")
                print(
                    f"=> Masses are evenly distributed from {self.Mbh_min} to {self.Mbh_max} solar masses"
                )

        elif self.mass_dis == "exp":
            Mbh_ave = self.Mbh_max - self.Mbh_min
            R1 = 1 - np.exp(-self.Mbh_min / Mbh_ave)
            R2 = 1 - np.exp(-self.Mbh_max / Mbh_ave)
            R = rnd.uniform(R1, R2, self.Nbh)
            masses = -Mbh_ave * np.log(1 - R)
            if verbose or v:
                print(f"Black Holes born correctly")
                print(f"=> You now own a population of {self.Nbh} Black holes")
                print(
                    f"=> Masses are exponentially distributed from {self.Mbh_min} to {self.Mbh_max} solar masses"
                )

        elif self.mass_dis == "kroupa":
            a = 2.3
            Mbh_unif = rnd.uniform(0, 1, self.Nbh)
            K = (1 - a) / (self.Mbh_max ** (1 - a) - self.Mbh_min ** (1 - a))
            Y = ((1 - a) / K * Mbh_unif + self.Mbh_min ** (1 - a)) ** (1 / (1 - a))
            jj = [(Y > self.Mbh_min) & (Y < self.Mbh_max)]
            masses = Y[tuple(jj)]
            if verbose or v:
                print(f"Black Holes born correctly")
                print(f"=> You now own a population of {self.Nbh} Black holes")
                print(
                    f"=> Masses are distributed with kroupa method from {self.Mbh_min} to {self.Mbh_max} solar masses"
                )

        elif self.mass_dis == "triang":
            masses = rnd.triangular(
                self.Mbh_min, (self.Mbh_max - self.Mbh_min) / 2, self.Mbh_max, self.Nbh
            )
            Mbh_ave = self.Mbh_max - self.Mbh_min
            if verbose or v:
                print(f"Black Holes born correctly")
                print(f"=> You now own a population of {self.Nbh} Black holes")
                print(
                    f"=> Masses are triangularly distributed from {self.Mbh_min} to {self.Mbh_max} solar masses"
                )

        # Populating the BH array with randomly extracted spins

        if self.spin_dis == "unif":
            spins = rnd.uniform(self.Sbh_min, self.Sbh_max, self.Nbh)
            if verbose or v:
                print(
                    f"=> Your Black Holes now have random spin uniformly distributed from {self.Sbh_min} to {self.Sbh_max}.\n"
                )

        elif self.spin_dis == "fixed":
            spins = np.array([self.Sbh_mean] * self.Nbh)
            if verbose or v:
                print(f"=> Your Black Holes now have equal spin of {self.Sbh_mean}.\n")

        elif self.spin_dis == "gauss":
            """
            It is worth noticing that if we were to simply extract values of the spin
            by using a gaussian distribution, we would have no easy way to exclude impossible values
            (e.g. if mean = .7 and sigma = .15, values >1 or <0 can be selected).
            Instead we prepare an array of values and randomly exctract from that.
            """
            step = (self.Sbh_max - self.Sbh_min) / self.Nbh
            _ = np.arange(self.Sbh_min, self.Sbh_max, step)
            gaussian = rnd.normal(self.Sbh_mean, self.Sbh_sigma, int(1e6))
            h, bin_edges = np.histogram(gaussian, bins=self.Nbh, density=True)
            p = h * np.diff(bin_edges)
            spins = rnd.choice(_, size=self.Nbh, p=p)
            if verbose or v:
                print(
                    f"=> Your Black Holes now have random spin with mean value {self.Sbh_mean}, a Gaussian distribution was used.\n"
                )

        self.Bhs = np.array([masses, spins])  # Storing the generated BHs into an array
        self.massesUM = masses  # This will not be used, actually.
        self.cluster_populated = True  # Updating cluster status

        if export_BH_data:
            # Returns an array of black holes masses and spin: k-th BH is Bhs[k][mass, spin]
            return self.Bhs

    def emit_GW(
        self,
        remove_undetectable=True,
        verbose=False,
        v=False,
        minimum_det_freq=20,
        maximum_det_freq=2048,
        tau_inst_mult=10,
    ):
        # This method is responsable of the emission of waves by the cluster

        if self.cluster_populated & self.mass_grid_built:
            if verbose or v:
                print("\nEmission of Gravitational Waves...")
                start = time()

            Mbhs = self.Bhs[0, :]  # Selecting masses only
            Sbhs = self.Bhs[1, :]  # Selecting spins only
            mus = self.boson_grid

            c_3 = self.c * self.c * self.c
            alpha = (
                self.G
                / (c_3 * self.hbar)
                * 2e30
                * Mbhs
                * mus[:, np.newaxis]
                * self.onev
            )

            # elevation to 9th power
            temp = alpha / 0.1
            for i in range(8):
                temp = temp * alpha / 0.1
            tau_inst = 27 * 86400 / 10.0 * Mbhs * (1 / temp) / Sbhs

            # elevetion to 15th pot
            temp = alpha / 0.1
            for i in range(14):
                temp = temp * alpha / 0.1

            tau_gw = 6.5e4 * 365 * 86400 * Mbhs / 10 * (1 / temp) / Sbhs
            del temp

            # =========================================
            # Calculating the emitted wave frequency
            # =========================================
            freq = (
                483
                * (
                    1
                    - 0.0056
                    / 8
                    * (Mbhs / 10.0) ** 2
                    * (mus[:, np.newaxis] / 1e-12) ** 2
                )
                * (mus[:, np.newaxis] / 1e-12)
            )

            if verbose or v:
                print(f"=> {freq.shape[0] * freq.shape[1]:.0E} Frequencies calculated")
                end = time()
                print(f"Total time for frequency calculation: {end - start:.2f} s")
                print(
                    f"Seconds per freq: {(end-start)/(freq.shape[0] * freq.shape[1]):.2E} s\n"
                )

            freq_max = (
                c_3
                / (2 * np.pi * self.G * 2e30 * Mbhs)
                * Sbhs
                / (1 + np.sqrt(1 - Sbhs**2))
            )  # ------ Maximum allowed GW frequency

            # elevation to 17th pot
            temp = alpha / 0.1
            for i in range(16):
                temp = temp * alpha / 0.1

            mus_2 = mus * mus / 1e-24
            fdot = (
                7e-15 * mus_2[:, np.newaxis] * temp
            )  # ----------------------------- Spin-up term due to boson annihilations

            fdot2 = (
                1e-10 * (10**17 / 1e30) ** 4 * mus_2[:, np.newaxis] * temp
            )  # ----------- Spin-up term due to boson emission
            fdot = fdot + fdot2
            del temp
            del fdot2

            # This section is a bit controversial, it has to do with the possibility of
            # depleting signals
            #
            # see https://github.com/Unoaccaso/Tesi-di-Laurea/blob/main/Riepilogo_Tesi.ipynb

            freq_now = freq + fdot * (self.cluster_eta_sec - tau_inst)

            dfdot = (
                self.Om0
                * np.sqrt(2 * np.ceil(freq_now / 10) * 10 * self.R0 / self.c)
                / (2 * self.Tobs / self.duty)
            )

            if verbose or v:
                print("\nCalculating wave amplitudes...")
                emission_start = time()

            chi_c = 4 * alpha / (1 + 4.0 * alpha * alpha)
            self.chi_c = chi_c

            temp = alpha / 0.1
            for i in range(6):
                temp = temp * alpha / 0.1
            h0 = (
                1 / np.sqrt(3) * 3.0e-24 / 10 * Mbhs * temp * (Sbhs - chi_c) / 0.5
            )  # --- GW peak amplitude at d=1 kpc
            del temp

            h0 = h0 / self.obs_distance

            timefactor = (
                1 + (self.cluster_eta_sec - tau_inst) / tau_gw
            )  # --------------------------- Time-dependent reduction factor
            h0 = h0 / timefactor

            del timefactor

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

            if verbose or v:
                emission_done = time()
                print(
                    f"=> Gravitational Waves emitted, elapsed time: {emission_done - emission_start:.2f} s"
                )

            if remove_undetectable:
                if verbose or v:
                    print("\nSelecting detectable Waves...")
                    start_selection = time()

                cond = np.array(
                    (tau_gw > 10 * self.cluster_eta_sec)
                    & (alpha < 0.1)
                    & (freq_now > 20)
                    & (freq_now < 2048)
                    & (tau_inst < 10 * self.cluster_eta_sec)
                    & (10 * tau_inst < tau_gw)
                    & (Sbhs > chi_c)
                    & (dfdot > fdot)
                )

                # cond = np.array(alpha < 0.1)

                # cond = 1

                # Applying conditions to detectable freqs
                tmp = freq_now * cond
                freq_now[tmp < 1] = np.nan

                self.tau_gw = tau_gw * cond
                self.tau_gw[self.tau_gw < 1] = np.nan
                self.tau_inst = tau_inst * cond
                self.tau_inst[self.tau_inst < 1] = np.nan
                self.chi_c = self.chi_c * cond
                self.chi_c[self.chi_c == 0] = np.nan
                self.alpha = alpha * cond
                self.alpha[self.alpha == 0] = np.nan
                self.saved_amps = h0 * cond
                self.saved_amps[self.saved_amps == 0] = np.nan

                if verbose or v:
                    print(
                        f"=> {self.n_mus - Mbhs.shape[0]} points were removed from the grid"
                    )
                # self.boson_grid = self.boson_grid[parser]
                if verbose or v:
                    selection_end = time()
                    print(
                        f"=> Grid Updated - elapsed time: {selection_end - start_selection:.2f} s"
                    )

            # Updating stored data
            if verbose or v:
                print("\nSaving data ...")
            """
            The code used leaves a 0 in places of BHs that didn't produce observable
            waves, by masking the arrays those values will not be considered in calculations.
            It is the fastest way to remove those data.
            """
            self.Bhs = np.array([Mbhs, Sbhs])
            self.saved_freqs = freq_now
            self.saved_freqs[self.saved_freqs < 0.1] = np.nan
            if verbose or v:
                print("=> Data saved\n\n")

        elif not self.cluster_populated:
            print(colored("==================  WARNING!  ==================", "red"))
            print(colored("No GW was emitted, cluster not populated.", "red"))
            print(colored("Run custer.populate() before trying to emit GWs.", "red"))
            print(colored("================================================", "red"))
        elif not self.mass_grid_built:
            print(
                colored("=====================  WARNING!  =====================", "red")
            )
            print(colored("No GW was emitted, mass grid not generated.", "red"))
            print(
                colored("Run custer.build_mass_grid() before trying to emit GWs", "red")
            )
            print(
                colored("======================================================", "red")
            )

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

    def calc_freq_distr(
        self,
        nsigma=1,
        nbins=32,
        norm_distr=True,
        remove_outliers=True,
        verbose=False,
        v=False,
    ):
        # Can be used to make an histogram with the frequencies calculated above

        if verbose or v:
            print("Calculating the frequency fluctuations...\n")
        sigma = {  # -------------------------------------------------------------------- Convert sigma to probability range
            1: 0.68,
            2: 0.95,
            3: 0.997,
        }
        sel_sigma = sigma[nsigma]
        freqs = np.sort(self.saved_freqs, kind="mergesort")  # sorting candidates
        masked_freqs = np.ma.masked_equal(freqs, 0)
        em_BHs = np.count_nonzero(freqs, axis=1)
        n_sel_BHs = np.around(em_BHs * sel_sigma).astype(int)

        # Creating the histogram of frequencies
        if verbose or v:
            print("Making the histograms...")

        bins, counts, binsizes = hist_by_row(
            masked_freqs, nbins=nbins, normalize=norm_distr
        )
        del masked_freqs  # Saving Memory space

        if verbose or v:
            print(
                "=> Saving histograms in:\n=> - cl.saved_hists_counts\n=> - cl.saved_hists_bins"
            )
        self.saved_hists_counts = counts
        self.saved_hists_bins = bins

        # Finding the most occurring frequency
        bin_mids = bins[:, :-1] + binsizes[:] / 2
        if verbose or v:
            print("=> Looking for peaks")
        max_freqs_idx = np.argmax(counts, axis=1)
        max_freqs = bin_mids[range(bin_mids.shape[0]), max_freqs_idx]

        # Looking for emitted frequency that is closer to the bin
        delta_freq = np.abs(freqs[:] - max_freqs[:, np.newaxis])
        closest_freq_to_bin_idx = np.argmin(delta_freq, axis=1)
        closest_freq_to_bin = freqs[range(freqs.shape[0]), closest_freq_to_bin_idx]

        # Finding indices of freqs inside the chosen sigma
        if verbose or v:
            print(f"=> Selecting frequencies in {sel_sigma * 100:.0f}% range of peak")
        max_idx = np.minimum(
            freqs.shape[1], closest_freq_to_bin_idx + np.around(n_sel_BHs / 2)
        )
        max_idx = max_idx.astype(int) + 1

        r = np.arange(freqs.shape[1])
        mask = (closest_freq_to_bin_idx[:, None] <= r) & (max_idx[:, None] >= 1)

        r_freq_in_sigma = np.full((freqs.shape[0], freqs.shape[1]), -1e15)
        r_freq_in_sigma[:][mask] = freqs[:][mask]

        leftovers = closest_freq_to_bin_idx + np.around(n_sel_BHs / 2) - freqs.shape[1]

        rem_indices = np.array(n_sel_BHs - np.around(n_sel_BHs / 2), dtype=int)
        rem_indices[leftovers > 0] = (
            rem_indices[leftovers > 0] + leftovers[leftovers > 0]
        )

        mask = (closest_freq_to_bin_idx[:, None] - rem_indices[:, None] <= r) & (
            closest_freq_to_bin_idx[:, None] >= 1
        )

        l_freq_in_sigma = np.full((freqs.shape[0], freqs.shape[1]), 1e15)
        l_freq_in_sigma[:][mask] = freqs[:][mask]

        if verbose or v:
            print("=> Calculating frequency variance")
        freq_min = l_freq_in_sigma.min(axis=1)
        freq_max = r_freq_in_sigma.max(axis=1)

        freqs_variance = freq_max - freq_min

        if remove_outliers:
            if verbose or v:
                print("\nRemoving outliers...")
            cond = freqs_variance < 20

            freqs_variance = freqs_variance[cond]
            max_freqs = max_freqs[cond]
            freq_max = freq_max[cond]
            freq_min = freq_min[cond]

            self.Bhs = np.array([self.Bhs[0][cond], self.Bhs[1][cond]])
            self.saved_freqs = self.saved_freqs[cond]
            self.saved_amps = self.saved_amps[cond]
            self.boson_grid = self.boson_grid[cond]
            self.saved_hists_bins = self.saved_hists_bins[cond]
            self.saved_hists_counts = self.saved_hists_counts[cond]

            if verbose or v:
                print(f"=> {cond[~cond].shape[-1]} points were removed")

        if verbose or v:
            print("\nSaving data and updating saved values...")
        self.saved_freqs_variance = freqs_variance
        self.saved_top_bins = max_freqs
        self.saved_rsigma_freq = freq_max
        self.saved_lsigma_freq = freq_min

        self.freq_distr_calculated = True
        if verbose or v:
            print("=> Data saved\n")

    def plot_freq_distr(self, mu, yscale="log", show_sigma=True, show_plot=True):
        # Used to plot the frequency distribution calculated

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

                if show_plot:
                    plt.bar(bins, counts, width=bin_size, align="edge")
                    plt.yscale(yscale)

                    plt.title(
                        f"Histogram of frequencies for boson mass of ${mu_value*1e12:.2f}$"
                        + "$\cdot 10^{-12} eV$"
                    )
                    plt.xlabel("Frequency $[Hz]$")
                    plt.ylabel("occ/prob")

                    if show_sigma:
                        plt.vlines(top_bin, 0, counts.max(), colors="black")
                        plt.vlines(rsigma, 0, counts.max())
                        plt.vlines(lsigma.max(), 0, counts.max())
                    plt.show()
                else:
                    return mu_value, mu, bins, counts, bin_size
            else:
                raise Exception("Please insert a reasonable value for boson mass")
        else:
            raise Exception(
                "Frequency distribution was not calculated, run cl.calc_freq_distr() before"
            )


def searchsorted_2d(a, v, side="left", sorter=None):
    # Make sure a and v are numpy arrays.
    a = np.asarray(a)
    v = np.asarray(v)

    # Augment a with row id
    ai = np.empty(a.shape, dtype=[("row", int), ("value", a.dtype)])
    ai["row"] = np.arange(a.shape[0]).reshape(-1, 1)
    ai["value"] = a

    # Augment v with row id
    vi = np.empty(v.shape, dtype=[("row", int), ("value", v.dtype)])
    vi["row"] = np.arange(v.shape[0]).reshape(-1, 1)
    vi["value"] = v

    # Perform searchsorted on augmented array.
    # The row information is embedded in the values, so only the equivalent rows
    # between a and v are considered.
    result = np.searchsorted(ai.flatten(), vi.flatten(), side=side, sorter=sorter)

    # Restore the original shape, decode the searchsorted indices so they apply to the original data.
    result = result.reshape(vi.shape) - vi["row"] * a.shape[1]

    return result


def hist_by_row(_, nbins=None, binsize=None, normalize=False, verbose=False, v=False):
    if ((nbins is None) and (binsize is None)) or (
        (nbins is not None) and (binsize is not None)
    ):
        raise Exception("Choose either a bin size or number of bins")

    bins = []
    counts = []
    bin_sizes = []
    if len(_.shape) < 2:
        if binsize is not None:
            _min = np.min(_)
            _max = np.max(_)
            if _min == _max:
                _max = _min + binsize

            nbins = int(np.ceil((_max - _min) / binsize))

        temp_counts, temp_bin = np.histogram(_[~_.mask], bins=nbins)

        if normalize:
            temp_counts = temp_counts / np.sum(~_.mask)

        counts.append(temp_counts)
        bins.append(temp_bin)
        bin_sizes.append(np.diff(temp_bin))

    else:
        for row in _:
            if binsize is not None:
                _min = np.min(row)
                _max = np.max(row)
                if _min == _max:
                    _max = _min + binsize

                nbins = int(np.ceil((_max - _min) / binsize))

            temp_counts, temp_bin = np.histogram(row[~row.mask], bins=nbins)

            if normalize:
                temp_counts = temp_counts / np.sum(~_.mask)

            counts.append(temp_counts)
            bins.append(temp_bin)
            bin_sizes.append(np.diff(temp_bin))

    if binsize is not None:
        l = max(map(len, bins))
        bins = np.array([np.append(x, [-1] * (l - len(x))) for x in bins])
        bins = np.ma.masked_where(bins < 0, bins)

        l = max(map(len, counts))
        counts = np.array([np.append(x, [-1] * (l - len(x))) for x in counts])
        counts = np.ma.masked_where(counts < 0, counts)

        l = max(map(len, bin_sizes))
        bin_sizes = np.array([np.append(x, [-1] * (l - len(x))) for x in bin_sizes])
        bin_sizes = np.ma.masked_where(bin_sizes < 0, bin_sizes)

        return bins, counts, bin_sizes
    else:
        return np.array(bins), np.array(counts), np.array(bin_sizes)
