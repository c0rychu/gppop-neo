import numpy as np


class Bin:
    def __init__(self, m_min, m_max):
        self.m_min = m_min
        self.m_max = m_max
        self.m_center = 0.5 * (m_min + m_max)
        self.log_m_log_center = 0.5 * (np.log(m_min) + np.log(m_max))

    def mask(self, m1):
        return (m1 >= self.m_min) & (m1 < self.m_max)

    @property
    def log_width(self):
        return np.log(self.m_max) - np.log(self.m_min)

    @property
    def width(self):
        return self.m_max - self.m_min


class Bins(list):
    def __init__(self, mbins):
        self.mbins = mbins
        for i in range(len(mbins)-1):
            self.append(Bin(m_min=mbins[i], m_max=mbins[i+1]))

    @property
    def m_centers(self):
        return np.array([bin.m_center for bin in self])

    @property
    def log_m_log_centers(self):
        return np.array([bin.log_m_log_center for bin in self])

    @property
    def log_widths(self):
        return np.array([bin.log_width for bin in self])

    @property
    def widths(self):
        return np.array([bin.width for bin in self])


class Piecewise:
    def __init__(self, bins, y):
        self.bins = bins
        self.y = y

    def __call__(self, x):
        return np.piecewise(x, [np.logical_and(x >= bin.m_min, x < bin.m_max) for bin in self.bins], self.y)  # noqa: E501
