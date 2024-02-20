import sys

sys.path.append("./")


from bosonClouds import sim_signals


if __name__ == "__main__":
    data = sim_signals(
        BH_mass_min=5,
        BH_mass_max=30,
        BH_mass_distribution="kroupa",
        N_BH_per_clouds=21000,
        boson_mass_min=1e-14,
        boson_mass_max=1e-11,
        boson_grid_type="geom",
        BH_age_distribution="logunif",
        BH_age_min_Yrs=float(4),
        BH_age_max_Yrs=float(6),
        N_bosons=1000,
    )

    generated_frequencies = data.frequency[0]
    generated_amplitudes = data.amplitude[0]

    data.inject_signals_and_CR(generated_frequencies, generated_amplitudes, 100)
