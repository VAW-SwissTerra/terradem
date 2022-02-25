# terradem --- Post-processing code for the TerrA image glacier reconstruction project.

## Structure
The `run.py` file shows what steps were performed in sequence to obtain the results.
Many steps were however performed non-sequentially due to revisions and final edits.
The sequence is therefore not a guarantee to fully reproduce the final data.

### Points of interest

- The code for all figures in the manuscript exist in `terradem/figures.py`.
- The code for the final glacier-wise and regional results is in `terradem/error.py::get_measurement_error()` (the error/uncertainty is evaluated and results are assimilated).

## Credit
The work was performed at [@VAW_glaciology](https://twitter.com/VAW_glaciology) in the frame of the work presented in Mannerfelt et al., ([2022; in discussion](https://doi.org/10.5194/tc-2022-14)).
It was co-financed by the Swiss Federal Office of Meteorology and Climatology ([MeteoSwiss](https://www.meteoswiss.admin.ch/)) in the frame of [GCOS Swizerland](https://www.meteoswiss.admin.ch/home/research-and-cooperation/international-cooperation/gcos/gcos-switzerland-projects.html).
