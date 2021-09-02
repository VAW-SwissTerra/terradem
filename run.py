"""Run the entire post-processing workflow."""
import os

import terradem.coregistration
import terradem.dem_tools
import terradem.error
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

    """
    for attribute in terradem.files.TERRAIN_ATTRIBUTES:
        terradem.dem_tools.generate_terrain_attribute(
            input_path=terradem.files.INPUT_FILE_PATHS["base_dem"],
            output_path=terradem.files.TEMP_FILES[f"base_dem_{attribute}"],
            overwrite=False,
            attribute=attribute,
        )

    """
    # At least 20% of the glaciers have to be covered by pixels for norm-regional-hypso
    min_coverage = 0.2
    print(f"Running inter-/extrapolation on scales: {terradem.files.INTERPOLATION_SCALES}")
    for scale in terradem.files.INTERPOLATION_SCALES:
        if scale == "national":
            warnings.warn("Skipping national scale")
            continue
        base_key = "ddem_coreg_tcorr_" + scale

        # Level -1 is national, 0 is SGI subregion 0, 1 is SGI subregion 1
        level = -1 if scale == "national" else int(scale.replace("subregion", ""))

        # These are the associated TEMP_FILES keys and output names for the interpolated products.
        interp_key = base_key + "-interp"
        interp_ideal_key = interp_key + "-ideal"

        # Interpolate the dDEM in the associated scale.
        print(f"Running normalized regional hypsometric interpolation on {scale} scale.")
        terradem.interpolation.subregion_normalized_hypsometric(
            ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
            output_filepath=terradem.files.TEMP_FILES[interp_key],
            output_filepath_ideal=terradem.files.TEMP_FILES[interp_ideal_key],
            level=level,
            min_coverage=min_coverage,
        )


        # These are the associated TEMP_FILES keys and output names for the extrapolated products.
        extrap_key = interp_key + "-extrap"
        extrap_ideal_key = extrap_key + "-ideal"

        # Extrapolate the interpolated dDEM in the associated scale.
        print(f"Running 'regular' regional hypsometric inter-/extrapolation on {scale} scale.")
        terradem.interpolation.regional_hypsometric(
            ddem_filepath=terradem.files.TEMP_FILES[interp_key],
            output_filepath=terradem.files.TEMP_FILES[extrap_key],
            output_filepath_ideal=terradem.files.TEMP_FILES[extrap_ideal_key],
        )
    """

    # terradem.error.compare_idealized_interpolation()

    # terradem.massbalance.get_volume_change()

    # Use the transforms obtained in the coregistration to transform the orthomosaics accordingly.
    # terradem.coregistration.transform_all_orthomosaics()

    # outline_differences = terradem.outlines.validate_outlines()

    # print(outline_differences.mean())


if __name__ == "__main__":
    main()
