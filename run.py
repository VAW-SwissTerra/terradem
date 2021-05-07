"""Run the entire post-processing workflow."""

import terradem.coregistration
import terradem.dem_tools


def main():

    # Perform bias correction and ICP coregistration on all DEMs to the base DEM
    # This takes a while! Probably because of bad threading locks to the shared data?
    terradem.coregistration.coregister_all_dems(n_threads=8)

    terradem.dem_tools.generate_ddems()
    terradem.dem_tools.merge_rasters(terradem.files.TEMP_SUBDIRS["ddems"], "temp/merged_ddem.tif")

    # Use the transforms obtained in the above coregistration to transform the orthomosaics accordingly.
    # terradem.coregistration.transform_orthomosaics()


if __name__ == "__main__":
    main()
