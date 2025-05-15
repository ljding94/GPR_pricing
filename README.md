# GPR Pricing Project

This repository implements numerical methods for pricing variance swap and  American put options along with sensitivity analyses. The code compute option prices and Greeks, as well as routines for generating datasets for calibration and machine learning studies.

## Directory Structure

- **analyze/**
  Contains analysis scripts such as [ML_analyze.py](analyze/ML_analyze.py) for exploring target distributions in the parameter space.

- **code/**
  Main pricing routines and dataset generation scripts:
  - [American_put.py](code/American_put.py)
  - [variance_swap.py](code/variance_swap.py)
  - [SVI_vol.py](code/SVI_vol.py)
  - [main.py](code/main.py)
    Contains the main function to run the pricing algorithms.
  - [main_test.py](code/main_test.py)
    Provides integration tests and visualizes some outputs (e.g. local volatility surface).

- **data/**
  Contains raw data files (tar.gz archives) and folders for test and pool data.

- **plot/**
  Contains visualization scripts:
  - [american_put_plot.py](plot/american_put_plot.py)
    Provides plotting routines for the American put and GPR fitting.
  - [vol_surface_plot.py](plot/vol_surface_plot.py)
    Uses matplotlib to plot volatility surfaces.

- **ref/**
  Contains additional resources and reference materials.

## Requirements

- Python 3.x
- NumPy
- SciPy
- matplotlib
- numdifftools
- (Optional) Other libraries as needed (e.g. for bicomplex arithmetic)

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   ```

2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   pip install -r requirements.txt
   ```

## Usage

- **Option Pricing:**
  Run pricing scripts under the `code` folder. For example, run the main test file:
  ```sh
  python code/main_test.py
  ```

- **Dataset Generation:**
  Use the dataset generation functions in [American_put.py](code/American_put.py) to create CSV files with option prices and greeks. Check the functions:
  - `generate_american_put_data_set`
  - `generate_american_put_precision_data`

- **Visualization:**
  Plot results using the scripts in the `plot` folder. For example, generate the American put GPR fitting plot by running the corresponding function in [american_put_plot.py](plot/american_put_plot.py).

## Files Overview

- **[code/American_put.py](code/American_put.py):**
  Implements the pricing functions using different numerical methods and dataset generation functions.

- **[plot/american_put_plot.py](plot/american_put_plot.py):**
  Contains plotting routines for visualizing the GPR fitting and simulation results.

- **[analyze/ML_analyze.py](analyze/ML_analyze.py):**
  Contains analysis routines to study the distribution of American put prices and greeks.
