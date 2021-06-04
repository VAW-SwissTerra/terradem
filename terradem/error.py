"""Functions to calculate DEM/dDEM error."""
from __future__ import annotations

import terradem.files


def compare_idealized_interpolation(
    idealized_ddem_path: str = terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp-ideal"],
    ddem_path: str = terradem.files.TEMP_FILES["ddem_coreg_tcorr"]
):
    """
    Compare an "idealized" interpolated dDEM with actual glacier values.

    :param idealized_ddem_path: The path to the idealized interpolated dDEM.
    :param ddem_path: The path to the dDEM without interpolation.
    """

    pass
