"""
Utility functions and classes for mock population / GPPOP study
Author: Cory Chu
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad


class SampleGenerator:
    def __init__(self, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def draw_samples(self, n=None):
        pass


class SampleGenerator1D(SampleGenerator):
    def __init__(self, unnormalized_pdf, x_min, x_max,
                 rng=None, log_space=True):
        super().__init__(rng)
        self._updf = unnormalized_pdf
        self._log_space = log_space
        self.x_min = x_min
        self.x_max = x_max

    def pdf(self, x):
        if not hasattr(self, "_pdf"):
            self._generate_pdf()
        return self._pdf(x)

    def cdf(self, x):
        if not hasattr(self, "_cdf"):
            self._generate_cdf()
        return self._cdf(x)

    def icdf(self, x):
        if not hasattr(self, "_icdf"):
            self._generate_icdf()
        return self._icdf(x)

    def draw_samples(self, N=None):
        return self.icdf(self.rng.random(N))

    def _generate_pdf(self):
        self._pdf_norm = quad(self._updf, self.x_min, self.x_max)[0]

        def pdf(x):
            prob = self._updf(x) / self._pdf_norm
            prob *= (x >= self.x_min) & (x < self.x_max)
            return prob

        self._pdf = pdf

    def _generate_ecdf(self):
        if self._log_space:
            self._xx = np.logspace(
                np.log10(self.x_min), np.log10(self.x_max), 1000)
        else:
            self._xx = np.linspace(self.x_min, self.x_max, 1000)
        self._ecdf = cumulative_trapezoid(self._updf(self._xx),
                                          self._xx,
                                          initial=0)
        self._ecdf /= self._ecdf[-1]

    def _generate_cdf(self):
        if not hasattr(self, "_ecdf"):
            self._generate_ecdf()
        self._cdf = lambda x: np.interp(x, self._xx, self._ecdf)

    def _generate_icdf(self):
        if not hasattr(self, "_ecdf"):
            self._generate_ecdf()
        self._icdf = lambda x: np.interp(x, self._ecdf, self._xx)


class SampleGeneratorM1geM2(SampleGenerator):
    def __init__(self,
                 m1_generator: SampleGenerator1D,
                 m2_generator: SampleGenerator1D,
                 rng=None):
        super().__init__(rng)
        self.m1_generator = m1_generator
        self.m2_generator = m2_generator

    def draw_samples(self, N=1):
        mass_1 = np.array([])
        mass_2 = np.array([])
        n_draw = N
        while (len(mass_1) < N):
            m1 = self.m1_generator.draw_samples(n_draw)
            m2 = self.m2_generator.draw_samples(n_draw)
            mask = m1 >= m2
            mass_1 = np.concatenate([mass_1, m1[mask]])
            mass_2 = np.concatenate([mass_2, m2[mask]])
            acceptance = (len(m1[mask])+1) / (n_draw+1)
            n_draw = int((N - len(mass_1)) / acceptance)
            # print("n_draw: {}, acceptance: {}".format(n_draw, acceptance))
        mass_1 = mass_1[0:N]
        mass_2 = mass_2[0:N]
        return mass_1, mass_2


class M1PowerLawWithGapM1M2Model(SampleGenerator):
    def __init__(self,
                 alpha,
                 beta,
                 m_min,
                 m_max,
                 gap_depth_ratio,
                 m_gap_min,
                 m_gap_max,
                 rng=None):
        super().__init__(rng)
        self.alpha = alpha
        self.beta = beta
        self.m_min = m_min
        self.m_max = m_max
        self.gap_depth_ratio = gap_depth_ratio
        self.m_gap_min = m_gap_min
        self.m_gap_max = m_gap_max
        self._create_draw_samples()

    def _create_draw_samples(self):
        def p_m1(m):
            return (Gap(self.gap_depth_ratio,
                        self.m_gap_min,
                        self.m_gap_max)(m)
                    * PowerLaw(self.alpha, self.m_min, self.m_max)(m))

        def p_m2(m):
            return (Gap(self.gap_depth_ratio,
                        self.m_gap_min,
                        self.m_gap_max)(m)
                    * PowerLaw(self.beta, self.m_min, self.m_max)(m))

        self.m1_generator = SampleGenerator1D(p_m1, self.m_min, self.m_max, rng=self.rng)  # noqa

        def draw_samples(n_draws=1):
            m1_samples = self.m1_generator.draw_samples(n_draws)

            m2 = []
            for m1 in m1_samples:
                m2_max = m1
                m2_min = self.m_min
                m2_generator = SampleGenerator1D(p_m2, m2_min, m2_max, rng=self.rng)  # noqa
                m2.append(m2_generator.draw_samples(1)[0])
            m2_samples = np.array(m2)

            return m1_samples, m2_samples
        self.draw_samples = draw_samples
    # TODO: add draw_samples method and m1_generator property explicitly.


class PowerLaw:
    """
    Power law distribution

    Parameters
    ----------
    alpha : float
        Power law index
    x_min : float, optional
        Minimum value of the distribution
    x_max : float, optional
        Maximum value of the distribution

    Methods
    -------
    __call__(x)
        Evaluate the distribution at x.
        If either x_min or x_max are not provided,
        returns x**alpha (unnormalized distribution).
        Otherwise returns the normalized distribution between x_min and x_max.

    Example
    -------
    p = PowerLaw(alpha=-2, x_min=1, x_max=10)
    p(x)
    """
    def __init__(self, alpha, x_min=None, x_max=None):
        self.alpha = alpha
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x):
        if self.x_min is None or self.x_max is None:
            return x**self.alpha
        else:
            """
            Based on powerlaw() in
            https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/utils.py
            """
            if self.alpha == -1:
                norm = 1 / np.log(self.x_max / self.x_min)
            else:
                norm = (1 + self.alpha) / \
                    (self.x_max ** (1 + self.alpha)
                     - self.x_min ** (1 + self.alpha))
            prob = x**self.alpha
            prob *= norm
            prob *= (x >= self.x_min) & (x < self.x_max)
            return prob


class Gap:
    """
    A mass gap distribution
    """
    def __init__(self,
                 gap_depth_ratio,
                 m_gap_min,
                 m_gap_max):
        self.gap_depth_ratio = gap_depth_ratio
        self.m_gap_min = m_gap_min
        self.m_gap_max = m_gap_max

    def __call__(self, m):
        in_gap = (m >= self.m_gap_min) & (m < self.m_gap_max)
        out_gap = np.invert(in_gap)
        return in_gap * self.gap_depth_ratio + out_gap * 1.0
