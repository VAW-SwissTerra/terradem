"""Run the entire post-processing workflow."""

import os

import terradem.coregistration
import terradem.dem_tools
import terradem.files
import terradem.utilities


def main():

    # Perform bias correction and ICP coregistration on all DEMs to the base DEM
    # This takes a while! Probably because of bad threading locks to the shared data?
    # terradem.coregistration.coregister_all_dems(n_threads=1)
    """
    terradem.dem_tools.generate_ddems(
        dem_filepaths=terradem.utilities.list_files(terradem.files.DIRECTORY_PATHS["dems"], r".*\.tif"),
        output_directory=terradem.files.TEMP_SUBDIRS["ddems_non_coreg"]
    )
    terradem.dem_tools.generate_ddems()
    """

    #ddem_selection = terradem.dem_tools.filter_ddems()

    #terradem.dem_tools.merge_rasters(ddem_selection, "temp/merged_ddem.tif")
    terradem.dem_tools.merge_rasters(
        terradem.utilities.list_files(terradem.files.TEMP_SUBDIRS["ddems_coreg_filtered"], r".*\.tif"),
        output_path=terradem.files.TEMP_FILES["ddem_coreg_filtered"]
    )

    # Use the transforms obtained in the above coregistration to transform the orthomosaics accordingly.
    # terradem.coregistration.transform_all_orthomosaics()


if __name__ == "__main__":
    main()
