# Eccentricity Detectability in EMRIs

This repository contains the numerical pipeline developed for the study of eccentricity detectability in EMRI systems, using a semi-coherent and multi-modal framework for their gravitational-wave signal analysis.

## Structure

- emri_tools/   → Core analysis code
- figures/      → Generated figures
- notebooks/    → Reproducible Jupyter notebooks for all plots
- populations/  → EMRI population datasets used in the study
- results/      → Cached outputs

## Dependencies

This project requires Python 3 and the following Python packages:

- [FastEMRIWaveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)
- [FastLISAResponse](https://github.com/BlackHolePerturbationToolkit/FastLISAResponse)
- [LISA Analysis Tools](https://github.com/BlackHolePerturbationToolkit/LISA-Analysis-Tools)
- NumPy
- SciPy
- multiprocess
- h5py
- pandas
- matplotlib

The analysis also uses tools from the
[EMRI Search repository](<EMRI-SEARCH-REPOSITORY-URL>).
