# ADDLICENSE


import numpy

SIMULATION = dict(
    N_BHS_IN_GALACTIC_CENTER=10_000,
    N_BHS_IN_HALO_CLUSTERS=100,
    N_CLUSTERS_IN_HALO=110,
    N_BOSONS=2000,
    GLOBULAR_CLUSTERS_DISTACE=30e3,  # Distance is in parsec
    BH_MASS_MIN=5,
    BH_MASS_MAX=30,
    BOSON_MASS_MIN=1e-14,
    BOSON_MASS_MAX=1e-11,
    SPIN_MIN=0.4,
    SPIN_MAX=0.6,
    # Ages
    HALO_BH_AGE_MIN=1e7,
    HALO_BH_AGE_MAX=1e9,
    CORE_BH_AGE_MIN=1e5,
    CORE_BH_AGE_MAX=1e7,
)
CONSTANTS = dict(
    # 1/day
    OM0=2 * numpy.pi / 86400,
    R0=5.5e6,  # Rotational radius at Livingston (lower latitude)
    ONEV=1.60217653e-19,
    FINT=1e30,  # Small interaction regime.
    DUTY=0.7,  # Detectors duty cycle (approximate)
    TOBS=365 * 86400 * 0.7,  # Here one should use the exact fraction of non-zero data,
)