from .base import Distribution
from .two_moons import TwoMoons
from .eight_gaussians import EightGaussians
from .checkerboard import Checkerboard
from .spirals import Spirals
from .concentric_rings import ConcentricRings
from .pinwheel import Pinwheel

DISTRIBUTIONS = {
    "two_moons": TwoMoons,
    "eight_gaussians": EightGaussians,
    "checkerboard": Checkerboard,
    "spirals": Spirals,
    "concentric_rings": ConcentricRings,
    "pinwheel": Pinwheel,
}

__all__ = [
    "Distribution", "TwoMoons", "EightGaussians", "Checkerboard",
    "Spirals", "ConcentricRings", "Pinwheel", "DISTRIBUTIONS"
]
