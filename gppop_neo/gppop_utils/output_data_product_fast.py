import numpy as np
from gppop.io import output_data_products
from numpy.typing import NDArray


class OutputDataProductsFast:
    """
    Speed up these functions:
        marginal_distributions_grid
        conditional_distributions_grid()
    provided in gppop.io.output_data_products

    Author: Cory Chu
    """

    def __init__(
        self, odp: output_data_products, grid_points_per_bin: int = 100
    ) -> None:
        """
        Parameters
        ----------
        odp : output_data_products
            An gppop.io.output_data_products object that contains the output data product.
        grid_points_per_bin : int
            The number of grid points per bin. Default is 100.
        """

        self.odp = odp
        self.num_bins_m_1dim = len(odp.mbins) - 1
        self.num_bins_z = odp.nbins_z
        self.num_bins_m_2dim = odp.nbins_m
        self.num_bins_tot = self.num_bins_m_2dim * self.num_bins_z

        idx_m1, idx_m2 = np.tril_indices(self.num_bins_m_1dim)
        self.idx_m1 = np.array(
            [
                idx_m1 + idx_z * self.num_bins_m_2dim
                for idx_z in np.arange(self.num_bins_z)
            ]
        ).ravel()
        self.idx_m2 = np.array(
            [
                idx_m2 + idx_z * self.num_bins_m_2dim
                for idx_z in np.arange(self.num_bins_z)
            ]
        ).ravel()
        self.idx_z = np.array(
            [
                np.zeros(self.num_bins_m_2dim) + idx_z
                for idx_z in np.arange(self.num_bins_z)
            ]
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
                np.linspace(
                    lo, hi, self.GRID_POINTS_PER_BIN
                )  # FIXME: should we add [:-1] here?
                for lo, hi in zip(self.odp.zbins[:-1], self.odp.zbins[1:])
            ]
        )

    def _get_sorting_idx_and_slices(
        self, idx: NDArray[np.int_]
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        A helper function for marg_flattened_matrix.

        Parameters
        ----------
        idx : NDArray[np.int_]
            The input index array.

        Returns
        -------
        sorting_idx : NDArray[np.int_]
            The index that would sort the input idx array.

        slices_of_sorting_idx : NDArray[np.int_]
            See Notes for an example.

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

    def marg_flattened_matrix(self, matrix, idx) -> NDArray[np.float_]:
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
        >>> marg_flattened_matrix(A, idx)
        array([[0, 5, 4],
               [1, 5, 4],
               [2, 5, 4],
               [5, 5, 4]])
        """
        sorting_idx, slices_of_sorting_idx = self._get_sorting_idx_and_slices(idx)
        return np.add.reduceat(matrix[..., sorting_idx], slices_of_sorting_idx, axis=-1)

    def get_marg_dist_log_m1_cond_on_z(self, reshape: bool = False) -> NDArray[np.float_]:
        """
        Marginalize the rates matrix over m2

        Parameters
        ----------
        reshape : bool, optional
            If True, reshape the output have the shape
            (num_mcmc_samples, num_bins_z, num_bins_m1).
            If False, the output will have the shape
            (num_mcmc_samples, num_bins_z * num_bins_m1).
            Default is False.

        Returns
        -------
        Rplogm1 : NDArray[np.float_]
            The marginalized rates matrix over m2, giving p(logm1, z).
            It is most likely to be a 2D array whose shape is
            (num_mcmc_samples, num_bins_z * num_bins_m1)
            where the last dimension is a flattened matrix from
            the 2D matrix with shape (num_bins_z, num_bins_m1).
        """
        matrix = self.odp.n_corr_samples * self.odp.delta_logm2s(self.odp.mbins)
        Rplogm1 = self.marg_flattened_matrix(matrix, self.idx_m1)
        if reshape:
            Rplogm1 = np.reshape(Rplogm1, (-1, self.num_bins_z, self.num_bins_m_1dim))
        return Rplogm1

    def get_marg_dist_log_m2_cond_on_z(self, reshape: bool = False) -> NDArray[np.float_]:
        """
        Marginalize the rates matrix over m1

        Parameters
        ----------
        reshape : bool, optional
            If True, reshape the output to have the shape
            (num_mcmc_samples, num_bins_z, num_bins_m2).
            If False, the output will have the shape
            (num_mcmc_samples, num_bins_z * num_bins_m2).
            Default is False.

        Returns
        -------
        Rplogm2 : NDArray[np.float_]
            The marginalized rates matrix over m1, giving p(logm2, z).
            It is most likely to be a 2D array whose shape is
            (num_mcmc_samples, num_bins_z * num_bins_m2)
            where the last dimension is a flattened matrix from
            the 2D matrix wtih shape (num_bins_z, num_bins_m2).
        """
        matrix = self.odp.n_corr_samples * self.odp.delta_logm1s(self.odp.mbins)
        Rplogm2 = self.marg_flattened_matrix(matrix, self.idx_m2)
        if reshape:
            Rplogm2 = np.reshape(Rplogm2, (-1, self.num_bins_z, self.num_bins_m_1dim))
        return Rplogm2

    def _get_marg_dist_m1_on_grid_at_lowest_z(self) -> NDArray[np.float_]:
        """
        Take the first z-slice from Rplogm1.
        Then, interpolate rates within each m1-bin to get p(m1|z = 1st z-bin)
        at the grid points of m1.

        Returns
        -------
        Rpm1 : NDArray[np.float_]
            It is most likely to be a 2-d array whose shape is
            (num_mcmc_samples, num_bins_m1 * (GRID_POINTS_PER_BIN-1))
        """
        Rplogm1 = self.get_marg_dist_log_m1_cond_on_z()
        Rplogm1 = Rplogm1[..., 0:self.num_bins_m_1dim]  # Take only the first z-slice
        Rpm1 = np.repeat(Rplogm1, self.GRID_POINTS_PER_BIN - 1, axis=-1) / self.ms
        return Rpm1

    def _get_marg_dist_m2_on_grid_at_lowest_z(self) -> NDArray[np.float_]:
        """
        Take the first z-slice from Rplogm2.
        Then, interpolate rates within each m2-bin to get p(m2|z = 1st z-bin)
        at the grid points of m2.

        Returns
        -------
        Rpm2 : NDArray[np.float_]
            It is most likely to be a 2D array whose shape is
            (num_mcmc_samples, num_bins_m2 * (GRID_POINTS_PER_BIN-1))
        """
        Rplogm2 = self.get_marg_dist_log_m2_cond_on_z()
        Rplogm2 = Rplogm2[..., 0:self.num_bins_m_1dim]  # Take only the first z-slice
        Rpm2 = np.repeat(Rplogm2, self.GRID_POINTS_PER_BIN - 1, axis=-1) / self.ms
        return Rpm2

    def get_marg_dist_z(self) -> NDArray[np.float_]:
        """
        Marginalize the rates matrix over m1 and m2.

        Returns
        -------
        Rz : NDArray[np.float_]
            The marginalized rates matrix over m1 and m2, giving p(z).
            It is most likely to be a 2D array whose shape is
            (num_mcmc_samples, num_bins_z)
        """
        log_bin_centers = self.odp.generate_log_bin_centers()
        diag_idx = np.where(log_bin_centers[:, 0] == log_bin_centers[:, 1])
        ones = np.ones(self.num_bins_tot)
        ones[diag_idx] *= 2  # TODO: Ask why do we need this.
        matrix = (
            self.odp.n_corr_samples
            * self.odp.delta_logm1s(self.odp.mbins)
            * self.odp.delta_logm2s(self.odp.mbins)
            * ones
        )
        Rz = self.marg_flattened_matrix(matrix, self.idx_z)
        return Rz

    def _get_marg_dist_z_on_grid(self) -> NDArray[np.float_]:
        """
        Interpolate rates within each z-bin to get p(z) at the grid points of z.

        Returns
        -------
        Rz : NDArray[np.float_]
            It is most likely to be a 2D array whose shape is
            (num_mcmc_samples, num_bins_z * GRID_POINTS_PER_BIN)
        """
        Rz = self.get_marg_dist_z()
        Rz = np.repeat(Rz, self.GRID_POINTS_PER_BIN, axis=-1)
        return Rz

    def _get_marg_dist_m1_on_grid_cond_on_z(self) -> NDArray[np.float_]:
        """
        Interpolate rates within each m1-bin to get p(m1|z) at the grid points of m1.

        Returns
        -------
        Rpm1 : NDArray[np.float_]
            It is most likely to be a 3D array whose shape is
            (num_mcmc_samples, num_bins_z, num_bins_m1 * (GRID_POINTS_PER_BIN-1))"""
        Rplogm1 = self.get_marg_dist_log_m1_cond_on_z(reshape=True)
        Rplogm1 = np.repeat(Rplogm1, self.GRID_POINTS_PER_BIN - 1, axis=-1)
        # FIXME: Can we eliminate the "-1" in self.GRID_POINTS_PER_BIN-1?
        Rpm1 = Rplogm1 / self.ms
        return Rpm1

    def _get_marg_dist_m2_on_grid_cond_on_z(self) -> NDArray[np.float_]:
        """
        Interpolate rates within each m2-bin to get p(m2|z) at the grid points of m2.

        Returns
        -------
        Rpm2 : NDArray[np.float_]
            It is most likely to be a 3D array whose shape is
            (num_mcmc_samples, num_bins_z, num_bins_m2 * (GRID_POINTS_PER_BIN-1))
        """
        Rplogm2 = self.get_marg_dist_log_m2_cond_on_z(reshape=True)
        Rplogm2 = np.repeat(Rplogm2, self.GRID_POINTS_PER_BIN - 1, axis=-1)
        # FIXME: Can we eliminate the "-1" in self.GRID_POINTS_PER_BIN-1?
        Rpm2 = Rplogm2 / self.ms
        return Rpm2

    # ------------------------------------------------------------------
    # Public API -------------------------------------------------------
    # ------------------------------------------------------------------

    def marginal_distributions_grid(
        self,
    ) -> tuple[
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
    ]:
        """
        Get the marginal distribution over z, and the marginal distributions
        at the first z-bin over m1 and m2 on the grid.
        When working on uncorrelated analysis, this gives:
        p(z), p(m1), and p(m2).

        Returns
        -------
        z : NDArray[np.float_]
            The grid points of z.
        Rz : NDArray[np.float_]
            p(z)
            with shape (num_mcmc_samples, len(z))
        m1 : NDArray[np.float_]
            The grid points of m1.
        Rpm1 : NDArray[np.float_]
            p(m1)
            with shape (num_mcmc_samples, len(m1))
        m2 : NDArray[np.float_]
            The grid points of m2.
        Rpm2 : NDArray[np.float_]
            p(m2)
            with shape (num_mcmc_samples, len(m2))
        """
        z = self.zs
        Rz = self._get_marg_dist_z_on_grid()
        m1 = self.ms
        Rpm1 = self._get_marg_dist_m1_on_grid_at_lowest_z()
        m2 = self.ms
        Rpm2 = self._get_marg_dist_m2_on_grid_at_lowest_z()
        return z, Rz, m1, Rpm1, m2, Rpm2

    def conditional_distributions_grid(
        self,
    ) -> tuple[
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
    ]:
        """
        Get p(m1|z) and p(m2|z) at the grid points of m1 and m2.

        Returns
        -------
        m1 : NDArray[np.float_]
            The grid points of m1.
        Rpm1 : NDArray[np.float_]
            p(m1|z)
            with shape (num_mcmc_samples, num_bins_z, len(m1))
        m2 : NDArray[np.float_]
            The grid points of m2.
        Rpm2 : NDArray[np.float_]
            p(m2|z)
            with shape (num_mcmc_samples, num_bins_z, len(m2))
        """
        m1 = self.ms
        Rpm1 = self._get_marg_dist_m1_on_grid_cond_on_z()
        m2 = self.ms
        Rpm2 = self._get_marg_dist_m2_on_grid_cond_on_z()
        return m1, Rpm1, m2, Rpm2
