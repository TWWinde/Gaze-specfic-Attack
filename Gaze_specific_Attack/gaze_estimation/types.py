import enum


class GazeEstimationMethod(enum.Enum):
    NVGaze = enum.auto()
    LPW = enum.auto()
    MPIIGaze = enum.auto()
    MPIIFaceGaze = enum.auto()
    GazeCapture = enum.auto()


class LossType(enum.Enum):
    L1 = enum.auto()
    L2 = enum.auto()
    SmoothL1 = enum.auto()
