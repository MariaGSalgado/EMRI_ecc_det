# Eccentricity Detectability in EMRIs

This repository contains the numerical pipeline developed for the study of eccentricity detectability in EMRI systems, using a semi-coherent framework for their gravitational-wave signal analysis.

## Structure

- emri_tools/   → Core analysis code
- figures/      → Generated figures
- notebooks/    → Reproducible Jupyter notebooks for all plots
- populations/  → EMRI population datasets used in the study
- results/      → Cached outputs

## Dependencies

This project requires Python 3 and the following Python packages:

- [FastEMRIWaveforms (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)
- [FastLISAResponse](https://github.com/mikekatz04/lisa-on-gpu/tree/master)
- [LISA Analysis Tools](https://github.com/lisa-analysis-tools/lisa-analysis-tools)
- NumPy
- SciPy
- multiprocess
- h5py
- pandas
- matplotlib

The analysis also uses tools from the
[EMRI Search repository](https://github.com/lorenzsp/EMRI-Search).

## Citation

If you use this code, please cite the paper: 

```bibtex
@misc{salgado2026eccentricitydetectabilityextrememassratio,
      title={Eccentricity detectability in Extreme Mass-Ratio Inspirals}, 
      author={Maria Gonçalves Salgado and Lorenzo Speri and Elena M. Rossi},
      year={2026},
      eprint={2611.15736},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2611.15736}, 
}
```
