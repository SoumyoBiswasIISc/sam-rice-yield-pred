# Claude AI Agent Project Guide

## Project Overview
* **Domain:** Agricultural Data Science
* **Objective:** Rice Yield Prediction
* **Goal:** To replicate Liu et al [2022] - "Rice Yield Prediction and Model Interpretation Based on Satellite and Climatic Indicators Using a Transformer Method" and to improve upon its results using new architectures/ideas

## Target Repository & Environment
* **Core Architecture:** Informer (AAAI 2021 Best Paper)
* **Source Repo:** https://github.com/zhouhaoyi/Informer2020
* **Strict Environment Constraints:**
  * Python Runtime: `3.6`
  * PyTorch Version: `1.8.0`
  * Core Stack: `matplotlib==3.1.1`, `numpy==1.19.4`, `pandas==0.25.1`, `scikit_learn==0.21.3`

## Dataset Tracking
Claude must expect, align, and preprocess the following 5 target datasets for yield prediction:
1. `MOD13A2` (Vegetation Indices / NDVI / EVI)
2. `CRUNCEP` (Climate / Temperature, Precipitation, Solar Radiation)
3. `CHIRPS` (High-resolution Precipitation/Rainfall Data)
4. `MCD12Q1` (Land Cover / Crop Type Masking)
5. `CSIF` (Solar-Induced Chlorophyll Fluorescence for photosynthetic activity)

## Data Alignment & Integrity Rules
* **Strict Temporal Alignment:** When processing and pairing satellite/climatic data, enforce an exact calendar-year match across all datasets.
* **LULC Masking Constraint:** Whenever handling `MOD13A2`, `CSIF`, `CHIRPS`, or `CRUNCEP` data from a specific year $X$, **ONLY** use the Land Use/Land Cover (LULC) mask from that exact same year $X$'s `MCD12Q1` data. 
* **Zero Cross-Year Mixing:** Never mix year $X$'s environmental metrics with another year $Y$'s `MCD12Q1` land cover mask.
* **Mandatory Metadata Inspection:** Always thoroughly inspect the metadata (spatial resolution, CRS, temporal frequency, nodata values, scaling factors, and band descriptions) of newly downloaded or raw datasets using tools like `rasterio`, `xarray`, or `gdal` *before* proceeding with any preprocessing or ingestion scripts.


## Code Style & Rules
* Prefer Python 3.6-compatible syntax (avoid modern features like match-case or strict post-3.6 type hinting features if they break older interpreters).
* Format code using standard PEP 8 guidelines.
* Explicitly comment data preprocessing steps, feature engineering, and model choices.
* **Reproducibility:** Always use `random_state=42` for data splits and stochastic models.
* **Data Leakage:** Enforce strict separation between train and validation/test pipelines.

## Directory Structure
* Root directory path: `/media/sam/writable/Sam Rice Yield Pred`
* Informer Source Code: `Informer2020/` (To be cloned)
* Data files: `data/`
* Model artifacts: `models/`
* Source scripts: `src/`

## Workspace Isolation & Constraints
* **Directory Jail:** You are strictly restricted to working within `/media/sam/writable/Sam Rice Yield Pred`. Do not attempt to read, write, navigate, or list files outside this directory path.
* **No Host Access:** Never execute bash or terminal commands that reference paths outside this project folder (e.g., do not touch `/home/sam/`, `/etc/`, or alternate drive mounts). All operations must remain local to the workspace.

## Project Assets & Data References
* **Target Districts List:** The definitive list of districts to be processed is located at:
  `/media/sam/writable/Sam Rice Yield Pred/district_index_map_IGP_LOYO_101_with_names.csv`
* **District Reference Schema:** When parsing this file, expect columns: `district_number`, `district_id`, `state_name`, `district_name`, `gid_0`, and `country`. 
* **Data Processing Filter:** Use the `district_id` (e.g., `IND.12.10_1`) or `district_name` (e.g., `Karnal`) from this file to filter and slice all spatial rasters and climatic datasets. Ignore any geographical data falling outside this district index mapping.

## Work Style
* Do not assume any important details missing from provided prompts or any details you're unable to find/reason out.
* Feel free to ask for these details before writing code.
* You can assume minor cosmetic details for convenience sake.

## Ground Truth File Schema
* **File Path:** `/media/sam/writable/Sam Rice Yield Pred/horizontal_crop_vertical_year_report (1).xls`
* **File Structure:** This file is raw HTML saved as `.xls`. It must be parsed using `pd.read_html()[0]`.
* **MultiIndex Column Layout:** The table has a 3-layer MultiIndex header:
  * Layer 0: `State`, `District`, `Year`, `Rice`
  * Layer 1: `State`, `District`, `Year`, `Kharif`
  * Layer 2: `State`, `District`, `Year`, `Area (Hectare)`, `Production (Tonnes)`, `Yield (Tonne/Hectare)`
* **Data Cleaning Rule:** State and District columns contain prefix numbering (e.g., "1. Haryana", "1. Ambala"). Always strip numbers and leading/trailing whitespace before attempting to match names.

## District Boundary & Name Alignment Rules
* **Shapefile Assets:** The definitive geographical boundary layers are loaded from the ESRI Shapefile component cluster at:
  `/media/sam/writable/Sam Rice Yield Pred/igp_101_districts_shp/igp_101_districts_shp.shp` (alongside its companion `.cpg`, `.dbf`, `.prj`, and `.shx` files).
* **Definitive Master Reference Index:** The master list of 101 target districts is explicitly tracked in:
  `/media/sam/writable/Sam Rice Yield Pred/district_index_map_IGP_LOYO_101_with_names.csv`
* **Mismatched Name Handling (Interactive Exception Strategy):** 
  * Do **NOT** use automated fuzzy string matching algorithms or make arbitrary assumptions for unmatched district names between the ground truth table and the master boundary `.csv`/`.shp` files.
  * For any district that fails a clean string match, the agent **MUST halt execution** or raise an interactive prompt detailing the unmatched name from the ground truth file alongside a list of available candidates from the boundary shapefile.
  * Prompt the user (`sam`) to manually map the name case-by-case.
  * Store these resolved manual overrides dynamically inside a localized lookup file (e.g., `district_name_mapping.json`) in the project root directory so you do not ask twice for the same district.

* **Strict 101-District Count Constraint:** 
  * The final processed pipeline dataset **MUST** contain exactly 101 unique districts matching the dimensions of your master index tracking file.
  * If the post-alignment dataset yields a count under 101 or over 101 unique districts, the agent **MUST halt execution** immediately.
  * Do not drop rows or inject dummy data automatically. Present the exact count discrepancy and the list of extra or missing districts to the user (`sam`), then ask for instructions on how to handle the mismatch.
