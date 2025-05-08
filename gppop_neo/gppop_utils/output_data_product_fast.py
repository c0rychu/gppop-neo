import numpy as np


class OutputDataProductsFast:
    """
    Speed up these functions:
        marginal_distributions_grid
        conditional_distributions_grid()
    provided in gppop.io.output_data_products

    Author: Yu-Kuang (Cory) Chu
    """
    def __init__(self, odp, grid_points_per_bin=100):
        self.odp = odp
        self.num_bins_m_1dim = len(odp.mbins) - 1
        self.num_bins_z = odp.nbins_z
        self.num_bins_m_2dim = odp.nbins_m
        self.num_bins_tot = self.num_bins_m_2dim * self.num_bins_z

        idx_m1, idx_m2 = np.tril_indices(self.num_bins_m_1dim)
        self.idx_m1 = np.array(
            [idx_m1 + idx_z * self.num_bins_m_2dim for idx_z in np.arange(self.num_bins_z)]
        ).ravel()
        self.idx_m2 = np.array(
            [idx_m2 + idx_z * self.num_bins_m_2dim for idx_z in np.arange(self.num_bins_z)]
        ).ravel()
        self.idx_z = np.array(
            [np.zeros(self.num_bins_m_2dim) + idx_z for idx_z in np.arange(self.num_bins_z)]
        ).ravel()
        self.GRID_POINTS_PER_BIN = grid_points_per_bin
        self.ms = np.concatenate(
            [
                np.linspace(lo, hi, self.GRID_POINTS_PER_BIN)[:-1]
                for lo, hi in zip(self.odp.mbins[:-1], self.odp.mbins[1:])
            ]
        )
        self.zs = np.concatenate(
            [
                np.linspace(lo, hi, self.GRID_POINTS_PER_BIN)  # FIXME: should we add [:-1] here?
                for lo, hi in zip(self.odp.zbins[:-1], self.odp.zbins[1:])
            ]
        )

    def _get_splits_and_order(self, idx):
        sorting_idx = np.argsort(idx)
        _, slices_of_sorting_idx = np.unique(idx[sorting_idx], return_index=True)
        return slices_of_sorting_idx, sorting_idx

    def _get_sorting_idx_and_slices(self, idx):
        """
        Notes
        -----
        idx = np.array([0, 2, 1, 2, 1, 1])
        sorting_idx = np.argsort(idx)  # [0, 2, 4, 5, 1, 3]
        sorted_idx = idx[sorting_idx]  # [0, 1, 1, 1, 2, 2]
                                          ^  ^        ^
                                          0  1        4
        So that the slices (0, 1), (1, 4), (4, END) are represented by a single array:
        slices_of_sorting_idx  # [0, 1, 4]
        """
        sorting_idx = np.argsort(idx)
        _, slices_of_sorting_idx = np.unique(idx[sorting_idx], return_index=True)
        return sorting_idx, slices_of_sorting_idx

    def _marg_flattened_matrix(self, matrix, idx):
        """
        Marginalize a flattened matrix by summing over elements with the same index.
        If the matrix is 2D, it assumes the last dimension the flattened matrix to work with.

        Examples
        --------
        >>> A = np.array([[0, 2, 1, 2, 3, 1],
        ...               [1, 2, 1, 2, 3, 1],
        ...               [2, 2, 1, 2, 3, 1],
        ...               [5, 2, 1, 2, 3, 1]])
        >>> idx = np.array([0, 2, 1, 2, 1, 1])
        >>> _marg_flattened_matrix(A, idx)
        array([[0, 5, 4],
               [1, 5, 4],
               [2, 5, 4],
               [5, 5, 4]])
        """
        sorting_idx, slices_of_sorting_idx = self._get_sorting_idx_and_slices(idx)
        return np.add.reduceat(matrix[..., sorting_idx], slices_of_sorting_idx, axis=-1)

    def _get_marg_dist_m1(self):
        matrix = self.odp.n_corr_samples * self.odp.delta_logm2s(self.odp.mbins)
        Rplogm1 = self._marg_flattened_matrix(matrix, self.idx_m1)
        Rplogm1 = Rplogm1[..., 0:self.num_bins_m_1dim]  # Take only the first z-slice
        Rpm1 = np.repeat(Rplogm1, self.GRID_POINTS_PER_BIN - 1, axis=-1) / self.ms
        return Rpm1

    def _get_marg_dist_m2(self):
        matrix = self.odp.n_corr_samples * self.odp.delta_logm1s(self.odp.mbins)
        Rplogm2 = self._marg_flattened_matrix(matrix, self.idx_m2)
        Rplogm2 = Rplogm2[..., 0:self.num_bins_m_1dim]  # Take only the first z-slice
        Rpm2 = np.repeat(Rplogm2, self.GRID_POINTS_PER_BIN - 1, axis=-1) / self.ms
        return Rpm2

    def _get_marg_dist_z(self):
        log_bin_centers = self.odp.generate_log_bin_centers()
        diag_idx = np.where(log_bin_centers[:, 0] == log_bin_centers[:, 1])
        ones = np.ones(self.num_bins_tot)
        ones[diag_idx] *= 2  # TODO: Ask why do we need this.
        matrix = (
            self.odp.n_corr_samples *
            self.odp.delta_logm1s(self.odp.mbins) *
            self.odp.delta_logm2s(self.odp.mbins) *
            ones
            )
        Rz = self._marg_flattened_matrix(matrix, self.idx_z)
        Rz = np.repeat(Rz, self.GRID_POINTS_PER_BIN, axis=-1)
        return Rz

    def marginal_distributions_grid(self):
        z = self.zs
        Rz = self._get_marg_dist_z()
        m1 = self.ms
        Rpm1 = self._get_marg_dist_m1()
        m2 = self.ms
        Rpm2 = self._get_marg_dist_m2()
        return z, Rz, m1, Rpm1, m2, Rpm2

    def _get_marg_dist_m1_cond_on_z(self):
        matrix = self.odp.n_corr_samples * self.odp.delta_logm2s(self.odp.mbins)
        Rplogm1 = self._marg_flattened_matrix(matrix, self.idx_m1)
        Rplogm1 = np.reshape(Rplogm1, (-1, self.num_bins_z, self.num_bins_m_1dim))
        Rplogm1 = np.repeat(Rplogm1, self.GRID_POINTS_PER_BIN - 1, axis=-1)
        # FIXME: Can we eliminate the "-1" in self.GRID_POINTS_PER_BIN-1?
        Rpm1 = Rplogm1 / self.ms
        return Rpm1

    def _get_marg_dist_m2_cond_on_z(self):
        matrix = self.odp.n_corr_samples * self.odp.delta_logm1s(self.odp.mbins)
        Rplogm2 = self._marg_flattened_matrix(matrix, self.idx_m2)
        Rplogm2 = np.reshape(Rplogm2, (-1, self.num_bins_z, self.num_bins_m_1dim))
        Rplogm2 = np.repeat(Rplogm2, self.GRID_POINTS_PER_BIN - 1, axis=-1)
        # FIXME: Can we eliminate the "-1" in self.GRID_POINTS_PER_BIN-1?
        Rpm2 = Rplogm2 / self.ms
        return Rpm2

    def conditional_distributions_grid(self):
        m1 = self.ms
        Rpm1 = self._get_marg_dist_m1_cond_on_z()
        m2 = self.ms
        Rpm2 = self._get_marg_dist_m2_cond_on_z()
        return m1, Rpm1, m2, Rpm2
