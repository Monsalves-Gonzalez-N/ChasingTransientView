# ChasingTransientView

Query galaxy catalogs within a sky localization region (gravitational wave, GRB, fast radio burst, etc.), compute the chance alignment probability for each source, and visualize results interactively with an ipyaladin widget.

```bash
conda env create -f datalab.yml
```
```bash
conda activate datalab && python query_localization.py --ra 155.5216 --dec -20.9972 --error 0.73arcmin
```

---

## Installation

### 1. Conda environment

```bash
conda env create -f datalab.yml
conda activate datalab
```

### 2. Key packages (installed via datalab.yml)

| Package | Purpose |
|---|---|
| `astro-datalab` | NOIRLab DataLab SQL client (`dl.queryClient`) |
| `astropy` | Sky coordinates, units, FITS I/O |
| `astroquery` | NED and VizieR (GLADE+) access |
| `ipyaladin` | Interactive Aladin Lite widget in Jupyter |
| `regions` | `CircleSkyRegion` for Aladin overlays |
| `numpy` / `pandas` | Array and catalog table handling |

---

## Usage

```bash
conda activate datalab
python query_localization.py --ra <RA> --dec <Dec> --error <radius>
```

The error radius accepts any astropy angle unit — with or without a space:

```bash
python query_localization.py --ra 155.5216 --dec -20.9972 --error 0.73arcmin
python query_localization.py --ra 47.2852  --dec 13.1544  --error 10arcmin
python query_localization.py --ra 133.015  --dec -66.007  --error 3arcmin
python query_localization.py --ra 155.5216 --dec -20.9972 --error 30arcsec
python query_localization.py --ra 155.5216 --dec -20.9972 --error 0.05deg
```

The script prints a source table to the terminal and opens a Jupyter Lab notebook with an interactive Aladin widget.

---

## Output

**Terminal table** — extended sources sorted by P_cc (4 significant figures):

| Column | Description |
|---|---|
| `offset_arcsec` | Angular separation from the transient |
| `mag_r` | r-band AB magnitude |
| `z_phot_mean` | Photometric redshift |
| `P_cc` | Chance alignment probability (Bloom et al. 2002) |

**Jupyter notebook** — Aladin widget with:
- Yellow circle marking the localization region
- Red markers for Legacy Survey sources (Legacy branch)
- Green markers for GLADE+ galaxies / purple for NED (fallback branch)

---

## Query strategy

1. **Legacy Survey DR10** (NOIRLab DataLab) — SQL cone search via `q3c_radial_query`, extended sources only (`type != 'PSF'`, `maskbits = 0`), joined with `photo_z` for photometric redshifts. Background survey: Legacy DR10 color.

2. **Fallback** (outside Legacy Survey footprint) — GLADE+ all-sky galaxy catalog via VizieR (`VII/291`), plus NED (120 s timeout). Background survey defaults to 2MASS color; commented options include Pan-STARRS, WISE, and Legacy DR10.

3. **NED** — available in both branches; active in the fallback notebook, commented out in the Legacy Survey notebook (use only for regions ≲ 10 arcmin).

---

## Chance Alignment Probability

Bloom et al. (2002), ApJ 572 L45:

```
P_cc = 1 − exp(−π × offset² × σ(<m_r))
σ(<m_r) = (1/3600) × 10^(0.33 × (m_r − 24))   [arcsec⁻²]
```

P_cc → 0: likely host galaxy. P_cc → 1: likely chance coincidence.
