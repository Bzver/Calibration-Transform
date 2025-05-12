# Calibration-Transform

This repository contains a simple python script that transforms the `sleap-anipose` calibration `.toml` file into `dannce` or `sdannce` complatible `.mat` files.

## Purpose

The main goal of this script is to provide a straightforward way to convert calibration data from `sleap-anipose` to `dannce` or `sdannce`. Specifically, it takes calibration files saved in the `.toml` format (generated in `sleap-anipose` workflows) and converts them into the `.mat` format expected by `Label3D` in `dannce` or `sdannce` workflow.

## How to Use

To use the script, follow these steps:

1.  **Clone the Repository**
2.  **Open the Script**
3.  **Change Project Directory:** Edit the script to specify the correct path to your project directory where the calibration file is located.
4.  **Run the Script:** Execute the script in your terminal or preferred Python environment.

## Requirements

Ensure you have the following set up before running the script:

* A `conda` environment with Python version **3.11 or above**.
* The following Python packages installed within your environment (along with their dependencies):
    * `matplotlib`
    * `scipy`
    * `numpy`
