# HBV_DA
HBV Data Assimilation Implementation
Author: Amirmoez Jamaat
Affiliation: Department of Civil and Environmental Engineering, The Pennsylvania State University
Overview
This repository contains the implementation of variational data assimilation (DA) methods for the differentiable HBV (δHBV) hydrological model, as described in the paper:
"Update hydrological states or meteorological forcings? Comparing data assimilation methods for differentiable hydrologic models"
Jamaat, A., Song, Y., Rahmani, F., Liu, J., & Shen, C. (2024)
The code implements three distinct data assimilation approaches:

HBV-DA-S: State-only adjustment
HBV-DA-P: Precipitation-only adjustment
HBV-DA-PS: Combined state and precipitation adjustment

Key Features
🎯 Three Assimilation Modes

State-Only Mode (state_only): Optimizes hydrological state variables (snowpack, soil moisture, meltwater, upper/lower zone storage) to minimize prediction errors
Precipitation-Only Mode (precip_only): Adjusts precipitation inputs using an optimizable multiplier (k⁴) to correct forcing data errors
State + Precipitation Mode (state_precip): Combines both approaches for maximum effectiveness

🔧 Technical Implementation

GPU-Accelerated: Leverages PyTorch for automatic differentiation and GPU computation
Batch Processing: Processes multiple basins simultaneously for efficiency
Flexible Time Configuration: Supports both calendar dates and day indices
Optimization: Uses Adadelta optimizer with configurable learning rates
Windowed Approach: 5-day assimilation window for optimal performance
