from enum import Enum

class NormalizationType(Enum):
    NO_NORMALIZATION = 1
    PERCENTAGE       = 2
    LOGARITHM        = 3
    SCALED_LOGARITHM = 4
