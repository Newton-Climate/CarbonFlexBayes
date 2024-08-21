# CarbonFlexBayes
## Author: Dr. Newton H. Nguyen
CarbonFlexBayes is a flexible carbon model that can be instantiated to have an arbitrary number of soil pools and layers. There is also integration with machine learning frameworks that can be switched in or defined by the user. Finally, The model can be constrained with observations and priors with multiple inversion algorithms implemented. This model uses Python for ease-of-use for the user.

# Software Features
## Model Features
- Soil carbon model can be initialized and run with arbitrary number of pools and layers through matrix approach.
- includes support for 13 and 14 carbon isotopes.
- sensativity and parameter analyses can be performed with automatic differentiation.
- optomization of parameters and posterior errors can be performed with constraints from flux, radio carbon, isotopic, and environmental observations.
- Specific processes e.g., soil turnover, transfer rate, microbial kinnetics, can be represented via machine learning.

## data Features
- Easy to use scripts to download any global climate model outputs and eddy flux data from NEON and Ameriflux.
- Re-grids and co-locates these data with radio carbon data from International Soil Radio Carbon Database (ISRaD) observations.
- Dates soil samples based on one or more soil pool models and atmospheric radio carbon time-series.

# Installation Instructions
In a Unix terminal, the following commands can be run to obtain the repository and build the dependencies.

```sh
git clone https://github.com/Newton-Climate/CarbonFlexBayes
cd CarbonFlexBayes

# Build the environment
conda env create -f environment.yml
conda activate carbon-flex-bayes# Installation Instructions
```

# Repository Contents
- Scripts: includes code that can be immediately run that will perform an action e.g., download and visualize data.
- src: includes files that contain functions to run models and perform analyses.

# Note
this repo is still being developed.
