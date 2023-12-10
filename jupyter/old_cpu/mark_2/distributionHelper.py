from numexpr import evaluate
from numpy.random import default_rng as rng
from numpy import (
    log,
    exp,
    diff,
    histogram,
    arange,
)


def kroupa(max, min, shape, a=2.3):
    a = 2.3
    zero_to_one = rng().uniform(
        low=0,
        high=1,
        size=shape,
    )
    M = max
    m = min
    K = (1 - a) / (M ** (1 - a) - m ** (1 - a))
    return evaluate("((1 - a) / K * zero_to_one + m ** (1 - a)) ** (1 / (1 - a))")


def uniform():
    return


def gaussian(max, min, shape):
    mean = (max + min) / 2
    step = (max - min) / shape[-1]
    _ = arange(min, max, step)
    gaussian = rng().normal(mean, 1, int(shape[-1] * 1e2))
    h, bin_edges = histogram(gaussian, bins=len(_), density=True)
    p = h * diff(bin_edges)
    return rng().choice(
        _,
        size=shape,
        p=p,
        shuffle=False,
    )


def exponential(max, min, shape):
    Mbh_ave = (max + min) / 2
    R1 = 1 - exp(-min / Mbh_ave)
    R2 = 1 - exp(-max / Mbh_ave)
    R = rng().uniform(
        low=R1,
        high=R2,
        size=shape,
    )
    return -Mbh_ave * log(1 - R)
