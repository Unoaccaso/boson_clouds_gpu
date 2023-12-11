# ADDLICENSE


import numpy
import cupy

SIMULATION = dict(
    N_BHS_IN_GALACTIC_CENTER=10_000,
    N_BHS_IN_HALO_CLUSTERS=100,
    N_CLUSTERS_IN_HALO=110,
    N_BOSONS=1_000,
    GLOBULAR_CLUSTERS_DISTACE=30,  # Distance is in kpc
    BH_MASS_MIN=5,
    BH_MASS_MAX=30,
    BOSON_MASS_MIN=1e-13,
    BOSON_MASS_MAX=2.5e-12,
    SPIN_MIN=0.4,
    SPIN_MAX=0.6,
    # Ages
    HALO_BH_AGE_MIN=10_000_000,
    HALO_BH_AGE_MAX=1_000_000_000,
    CORE_BH_AGE_MIN=100_000,
    CORE_BH_AGE_MAX=10_000_000,
)
#! THIS DON'T WORK, VALUES ARE HARD CODED NOW!!!
CONSTANTS = dict(
    # 1/day
    OM0=2 * numpy.pi / 86400,
    R0=5.5e6,  # Rotational radius at Livingston (lower latitude)
    ONEV=1.60217653e-19,
    FINT=1e30,  # Small interaction regime.
    DUTY=0.7,  # Detectors duty cycle (approximate)
    TOBS=365 * 86400 * 0.7,  # Here one should use the exact fraction of non-zero data,
)

GENERAL = dict(
    PRECISION=cupy.float32,
)

CUDA = dict(
    RTX_3050TI=dict(
        MULTIPROCESSORS=20,  # RT CORES
        MAX_THREAD_PER_BLOCK=1024,
        MAX_THREAD_PER_RT=1536,
        MAX_DIM_BLOCK=(1024, 1024, 64),
    ),
    TESLA_K20=dict(
        MULTIPROCESSORS=13,
        MAX_THREAD_PER_BLOCK=1024,
        MAX_THREAD_PER_RT=2048,
        MAX_DIM_BLOCK=(1024, 1024, 64),
    ),
    BLOCK_SIZE=(32, 8),  # (x, y)
)
