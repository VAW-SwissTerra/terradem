"""Run the entire post-processing workflow."""
import pandas as pd

import terradem.coregistration
import terradem.dem_tools
import terradem.files
import terradem.interpolation
import terradem.massbalance
import terradem.outlines
import terradem.utilities


def main():

    # Perform bias correction and ICP coregistration on all DEMs to the base DEM
    # This takes a while! Probably because of bad threading locks to the shared data?
    """
    terradem.coregistration.coregister_all_dems(n_threads=1)
    # Generate dDEMs for the coregistered DEMs
    terradem.dem_tools.generate_ddems()

    # Generate temporally corrected dDEMs
    terradem.dem_tools.ddem_temporal_correction()

    # Generate dDEMs for the non-coregistered DEMs
    terradem.dem_tools.generate_ddems(
        dem_filepaths=terradem.utilities.list_files(terradem.files.DIRECTORY_PATHS["dems"], r".*\.tif"),
        output_directory=terradem.files.TEMP_SUBDIRS["ddems_non_coreg"]
    )
    terradem.dem_tools.get_ddem_statistics()
    """
    # ddem_selection = terradem.dem_tools.filter_ddems()

    # terradem.dem_tools.merge_rasters(ddem_selection, "temp/merged_ddem.tif")
    """
    terradem.dem_tools.merge_rasters(
        terradem.utilities.list_files(terradem.files.TEMP_SUBDIRS["ddems_coreg_tcorr"], r".*\.tif"),
        output_path=terradem.files.TEMP_FILES["ddem_coreg_tcorr"]
    )
    """

    """
    terradem.interpolation.normalized_regional_hypsometric(
        terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp"],
        terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp_signal"],
        signal=signal
    )
    """

    terradem.interpolation.get_regional_signals(level=0)

    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion0-interp"],
        level=0,
        min_coverage=0.1
    )

    terradem.massbalance.get_volume_change()

    # Use the transforms obtained in the coregistration to transform the orthomosaics accordingly.
    # terradem.coregistration.transform_all_orthomosaics()

    #outline_differences = terradem.outlines.validate_outlines()

    # print(outline_differences.mean())


if __name__ == "__main__":
    main()
