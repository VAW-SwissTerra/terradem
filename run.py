"""Run the entire post-processing workflow."""
import os

import terradem.coregistration
import terradem.dem_tools
import terradem.files
import terradem.interpolation
import terradem.massbalance
import terradem.outlines
import terradem.utilities


def main() -> None:

    # This is the only dDEM that was integral to remove but wasn't removed automatically, so here I am, hardcoding...
    if os.path.isfile(os.path.join(terradem.files.TEMP_SUBDIRS["ddems_coreg_tcorr"], "station_2249_ddem.tif")):
        os.remove(os.path.join(terradem.files.TEMP_SUBDIRS["ddems_coreg_tcorr"], "station_2249_ddem.tif"))  # noqa

    # Perform bias correction and ICP coregistration on all DEMs to the base DEM
    # This takes a while! Probably because of bad threading locks to the shared data?
    r"""
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

    terradem.dem_tools.merge_rasters(
        terradem.utilities.list_files(terradem.files.TEMP_SUBDIRS["ddems_coreg_tcorr"], r".*\.tif$"),
        output_path=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        min_median=-350 / 90,
        max_median=100 / 90,
    )

    # Create raster versions of each SGI subregion
    terradem.outlines.rasterize_sgi_zones(level=0, overwrite=False)
    terradem.outlines.rasterize_sgi_zones(level=1, overwrite=False)

    print("Extracting regional signals.")
    terradem.interpolation.get_regional_signals(level=0)
    terradem.interpolation.get_regional_signals(level=1)

    # At least 20% of the glaciers have to be covered by pixels for norm-regional-hypso
    min_coverage = 0.2

    print("Running normalized regional hypsometric interpolation (without subregions).")
    terradem.interpolation.normalized_regional_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp"],
        output_signal_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp_signal"],
        min_coverage=min_coverage,
    )

    print("Running normalized regional hypsometric interpolation (subregion level 0).")
    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion0-interp"],
        level=0,
        min_coverage=min_coverage,
    )

    print("Running normalized regional hypsometric interpolation (subregion level 1).")
    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion0-interp"],
        level=1,
        min_coverage=min_coverage,
    )
    """
    terradem.dem_tools.generate_terrain_attribute(
        input_path=terradem.files.INPUT_FILE_PATHS["base_dem"],
        output_path=terradem.files.TEMP_FILES["base_dem_slope"],
        attribute="slope",
    )

    terradem.dem_tools.generate_terrain_attribute(
        input_path=terradem.files.INPUT_FILE_PATHS["base_dem"],
        output_path=terradem.files.TEMP_FILES["base_dem_aspect"],
        attribute="aspect",
    )
    terradem.dem_tools.generate_terrain_attribute(
        input_path=terradem.files.INPUT_FILE_PATHS["base_dem"],
        output_path=terradem.files.TEMP_FILES["base_dem_curvature"],
        attribute="curvature",
    )
    """

    # Generate idealized dDEMs to compare the discrepancy between them and the actual data.
    print("Generating idealized dDEM (without subregions)")
    terradem.interpolation.normalized_regional_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp-ideal"],
        output_signal_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp_signal"],
        signal=terradem.interpolation.read_hypsometric_signal(
            terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp_signal"]
        ),
        idealized_ddem=True,
        min_coverage=min_coverage,
    )

    print("Generating idealized dDEM (subregion level 0)")
    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion0-interp-ideal"],
        level=0,
        min_coverage=min_coverage,
        idealized_ddem=True,
    )

    print("Generating idealized dDEM (subregion level 1)")
    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion-interp-ideal"],
        level=1,
        min_coverage=min_coverage,
        idealized_ddem=True,
    )

    # terradem.massbalance.get_volume_change()

    # Use the transforms obtained in the coregistration to transform the orthomosaics accordingly.
    # terradem.coregistration.transform_all_orthomosaics()

    # outline_differences = terradem.outlines.validate_outlines()

    # print(outline_differences.mean())


if __name__ == "__main__":
    main()
