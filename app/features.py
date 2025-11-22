import numpy as np
from state import buf


def extract_features_from_buf() -> tuple[int, int, bool]:
    if len(buf) == 0:
        return 0, 0, False
    raw_extractions = np.array(buf, dtype=float)
    medians = np.median(raw_extractions, axis=0)
    person_med = medians[0]
    luggage_med = medians[1]
    hawaii_med = medians[2] > 0.75
    return person_med, luggage_med, hawaii_med
