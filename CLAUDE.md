# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses a conda environment defined in `datalab.yml`:

```bash
conda env create -f datalab.yml
conda activate datalab
```

To launch Jupyter:
```bash
jupyter lab
# or
jupyter notebook
```

## Project Purpose

Programmatic access to astronomical objects within sky localization regions (e.g., gravitational wave or gamma-ray burst error boxes). The workflow queries galaxy/source catalogs to find counterpart candidates given a sky position (RA, Dec) and localization error radius.

## Architecture

All work lives in `aladin.ipynb`, a single Jupyter notebook with two complementary strategies:

**Part 1 — Legacy Survey DR10** (primary, for well-covered sky regions):
- Authenticates to NOIRLab DataLab via `dl.authClient`
- Runs SQL against `ls_dr10.tractor` and `ls_dr10.photo_z` tables using `dl.queryClient.query()`
- Spatial filtering uses `q3c_radial_query(ra, dec, target_ra, target_dec, radius_deg)`
- Filters out stars (`type != 'PSF'`) and masked regions (`maskbits = 0`)
- Joins photometric redshifts from `photo_z` table
- Visualizes with `ipyaladin` (interactive Aladin Lite widget)

**Part 2 — NED + All-Sky Surveys** (fallback for regions outside Legacy Survey footprint):
- Queries NASA/IPAC Extragalactic Database via `astroquery.ipac.ned.Ned.query_region()`
- Same `ipyaladin` visualization with `CircleSkyRegion` overlays

## Key Parameters

The core inputs for each query are:
- `ra`, `dec` — target coordinates in decimal degrees (ICRS)
- `loc_error` — localization radius as an `astropy.units.Quantity` (e.g., `0.73 * u.arcmin`)
- `radius_deg` — derived from `loc_error.to(u.deg).value`

## Available Survey Backgrounds for Aladin

Commented survey strings in the notebook (set via `aladin.survey = "..."`):
- `"CDS/P/DESI-Legacy-Surveys/DR10/color"` — Legacy Survey DR10
- `"CDS/P/PanSTARRS/DR1/color-i-r-g"` — Pan-STARRS optical
- `"CDS/P/2MASS/color"` — 2MASS near-IR
- `"CDS/P/allWISE/color"` — WISE mid-IR
