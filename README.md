# Disrupted basal forebrain-cortical functional topography predicts cognitive decline in Parkinson's disease

This repository contains code supporting the basal forebrain functional-gradient, normative cortical-expression, molecular PET correspondence, and cortical geodesic-distance analyses.

The basal forebrain gradient workflow was adapted from:

Chakraborty S, Haast RAM, Onuska KM, et al. Multimodal gradients of basal forebrain connectivity across the neocortex. Nature Communications. 2024;15:8990. [doi:10.1038/s41467-024-53148-x](https://doi.10.1038/s41467-024-53148-x).

The analyses also make use of [BrainSpace](https://brainspace.readthedocs.io/en/latest/), [neuromaps](https://netneurolab.github.io/neuromaps/index.html), the [Schaefer cortical parcellation](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal) and the [Geodesic Library](https://github.com/the-virtual-brain/tvb-gdist) (gdist). Please cite the corresponding software, atlases, and datasets when reusing this code.

## Overview

Resting-state functional connectivity between basal forebrain voxels and 400 cortical parcels is used to derive a continuous low-dimensional gradient of basal forebrain-cortical functional organisation.

### The analysis

- derives a group-level basal forebrain functional gradient using diffusion map embedding
- aligns participant-level gradients to the group reference using Procrustes alignment
- generates gradient-weighted cortical expression maps
- derives a healthy-control normative cortical expression pattern
- quantifies individual similarity to this normative pattern
- evaluates spatial correspondence with independent molecular PET maps using spin permutation testing
- calculates cortical geodesic distance from surface-projected basal forebrain cholinergic subdivisions.

The primary individual similarity measure is the Spearman rank correlation between each participant's gradient-weighted cortical expression map and the healthy-control normative cortical expression pattern. Lower similarity indicates greater deviation from normative basal forebrain-cortical functional organisation.

### 1. Basal forebrain functional gradients

BF_func_gradients.ipynb

The notebook contains the main basal forebrain gradient workflow and similarity to normative cortical expression of gradient.

### 2. Spin permutations

make_spins.py

Generates parcel-wise spatial rotations for spin permutation testing using the Alexander-Bloch method implemented in neuromaps.

The script generates 10,000 rotations of the Schaefer 400-region parcellation on the fsaverage 10k surface and converts the resulting indices to a zero-based 400-parcel scheme.

Outputs:

- spins_400x10000.npy
- spins_400x10000.mat

### 3. Molecular PET correspondence

PET_spintest_correspondence.py

Tests spatial correspondence between the healthy-control normative cortical expression of the basal forebrain gradient and independent molecular PET maps from neuromaps.

The primary cholinergic analysis uses three independent [18F]-FEOBV vesicular acetylcholine transporter (VAChT) datasets to generate a consensus VAChT map.

Additional PET maps can be included as specificity analyses.

### 4. Cortical geodesic distance

geodesic_distance.py

Calculates cortical geodesic distance from surface-projected basal forebrain seeds.

Outputs include parcel-wise geodesic distances and optional vertex-wise GIFTI maps.

### Resources

Analysis-specific basal forebrain resources:

- BF_seed_2mm.nii.gz
- BF_masked_fullBF_2mm.dseg.nii.gz
- lh.BF_Ch123.fsavg5.shape.gii
- rh.BF_Ch123.fsavg5.shape.gii
- lh.BF_Ch4.fsavg5.shape.gii
- rh.BF_Ch4.fsavg5.shape.gii

Whole-basal-forebrain surface projections used for notebook visualisation:

- seed-BASF.L.bin.fsa5.shape.gii
- seed-BASF.R.bin.fsa5.shape.gii

Additional standard resources required by the scripts include:

- fsaverage5 / fsaverage 10k cortical surfaces
- left- and right-hemisphere Schaefer 400-region annotation files


### Software

Analyses were developed in Python 3.11.

Core dependencies include:

- numpy
- pandas
- scipy
- h5py
- nibabel
- nilearn
- scikit-learn
- brainspace
- neuromaps
- surfplot
- statsmodels
- pingouin
- matplotlib
- seaborn
- gdist
- openpyxl

Exact package versions are available in requirements.txt 

### Data availability

Participant-level imaging and clinical data were obtained from the Parkinson's Progression Markers Initiative (PPMI) and are openly available to researchers upon request (https://www.ppmi-info.org/access-data-specimens/data). 
