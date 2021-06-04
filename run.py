"""Run the entire post-processing workflow."""
import terradem.coregistration
import terradem.dem_tools
import terradem.files
import terradem.interpolation
import terradem.massbalance
import terradem.outlines
import terradem.utilities


def main() -> None:

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
    # ddem_selection = terradem.dem_tools.filter_ddems()

    # terradem.dem_tools.merge_rasters(ddem_selection, "temp/merged_ddem.tif")
    r"""
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

    """
    terradem.outlines.rasterize_sgi_zones(level=0, overwrite=False)

    print("Extracting regional signals.")
    terradem.interpolation.get_regional_signals(level=0)

    print("Interpolating dDEM.")
    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion0-interp"],
        level=0,
        min_coverage=0.1
    )
    """

    print("Generating idealized dDEMs")
    terradem.interpolation.normalized_regional_hypsometric(
        terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp"],
        terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp_signal"],
        signal=terradem.interpolation.read_hypsometric_signal(
            terradem.files.TEMP_FILES["ddem_coreg_tcorr_interp_signal"]
        ),
    )
    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion0-interp-ideal"],
        level=0,
        min_coverage=0.1,
        idealized_ddem=True,
    )
    terradem.interpolation.subregion_normalized_hypsometric(
        ddem_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr"],
        output_filepath=terradem.files.TEMP_FILES["ddem_coreg_tcorr_subregion-interp-ideal"],
        level=1,
        min_coverage=0.1,
        idealized_ddem=True,
    )

    # terradem.massbalance.get_volume_change()

    # Use the transforms obtained in the coregistration to transform the orthomosaics accordingly.
    # terradem.coregistration.transform_all_orthomosaics()

    # outline_differences = terradem.outlines.validate_outlines()

    # print(outline_differences.mean())


if __name__ == "__main__":
    main()
