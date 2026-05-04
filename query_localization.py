#!/usr/bin/env python
"""
query_localization.py
Query galaxy catalogs within a sky localization region, open an interactive
Aladin Lite view with Legacy Survey background, and compute the chance
alignment probability for each source.

Usage:
    python query_localization.py --ra 155.5216 --dec -20.9972 --error 0.73arcmin
    python query_localization.py --ra 133.015  --dec -66.007  --error 3arcmin
"""

import argparse
import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_error(error_str: str) -> u.Quantity:
    """Parse a localization-error string using astropy, e.g. '0.73arcmin', '30arcsec', '1.5deg'."""
    try:
        q = u.Quantity(error_str.strip())
        return q.to(u.deg).to(q.unit)   # validates it's an angle unit
    except Exception:
        raise ValueError(
            f"Cannot parse error {error_str!r}. "
            "Expected format like '0.73arcmin', '30 arcsec', or '1.5deg'."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query galaxy catalogs within a sky localization region.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ra",    type=float, required=True, help="RA in decimal degrees (ICRS)")
    p.add_argument("--dec",   type=float, required=True, help="Dec in decimal degrees (ICRS)")
    p.add_argument("--error", type=str,   required=True,
                   help="Localization radius with unit, e.g. '0.73arcmin' or '30arcsec'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chance Alignment Probability  (Bloom et al. 2002, ApJ 572 L45)
# ---------------------------------------------------------------------------

def _mag_from_nanomaggy(flux: np.ndarray) -> np.ndarray:
    """Convert Legacy Survey r-band flux in nanomaggies to AB magnitude."""
    flux = np.asarray(flux, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(flux > 0, 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan)), np.nan)
    return mag


def chance_alignment_prob(
    offset_arcsec: np.ndarray,
    mag_r: np.ndarray,
    loc_error_arcsec: float,
) -> np.ndarray:
    """
    Bloom et al. (2002) chance coincidence probability.

        P_cc = 1 - exp(-pi * r^2 * sigma(<m_r))

    Effective radius (accounts for localization error):
        r = loc_error_arcsec           if loc_error > offset
        r = sqrt(offset^2 + err^2)     otherwise

    Surface density (Bloom 2002 eq. 3, r-band):
        sigma(<m_r) = 10^(0.334*(m_r - 22.963) + 4.32) / (3600^2 * 0.334 * ln10)

    Parameters
    ----------
    offset_arcsec    : angular separation from the transient (arcsec)
    mag_r            : r-band AB magnitude of the galaxy
    loc_error_arcsec : 1-sigma localization radius (arcsec)

    Returns
    -------
    P_cc in [0, 1]  (NaN when magnitude is undefined)
    """
    offset = np.asarray(offset_arcsec, dtype=float)
    mag_r  = np.asarray(mag_r, dtype=float)
    err    = float(loc_error_arcsec)

    r     = np.where(err > offset, err, np.sqrt(offset**2 + err**2))
    norm  = (3600.0**2) * 0.334 * np.log(10.0)
    sigma = 10.0 ** (0.334 * (mag_r - 22.963) + 4.32) / norm
    p_cc  = 1.0 - np.exp(-np.pi * r**2 * sigma)
    return np.where(np.isfinite(p_cc), np.clip(p_cc, 0.0, 1.0), np.nan)


# ---------------------------------------------------------------------------
# Legacy Survey DR10  (NOIRLab DataLab)
# ---------------------------------------------------------------------------

def query_legacy(ra: float, dec: float, loc_error: u.Quantity) -> pd.DataFrame:
    """
    Cone search against ls_dr10.tractor + photo_z via NOIRLab DataLab.
    Returns a DataFrame with columns including offset_arcsec, mag_r, and P_cc.
    """
    try:
        import dl.queryClient as qc  # type: ignore
    except ImportError:
        print("[WARNING] dl package not installed; skipping Legacy Survey query.")
        return pd.DataFrame()

    radius_deg = loc_error.to(u.deg).value
    query = f"""
WITH cone AS (
    SELECT ls_id, ra, dec, type, flux_r, mw_transmission_r
    FROM ls_dr10.tractor
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})
      AND type != 'PSF'
      AND maskbits = 0
)
SELECT c.ls_id, c.ra, c.dec, c.type, c.flux_r, c.mw_transmission_r,
       p.z_phot_mean, p.z_spec, p.z_phot_std
FROM cone c
JOIN ls_dr10.photo_z p ON c.ls_id = p.ls_id
WHERE p.z_phot_mean IS NOT NULL;
"""
    try:
        df = qc.query(sql=query, fmt="pandas")
    except Exception as e:
        print(f"[WARNING] Legacy Survey query failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Angular offset from transient
    target  = SkyCoord(ra, dec, unit="deg", frame="icrs")
    sources = SkyCoord(df["ra"].values, df["dec"].values, unit="deg", frame="icrs")
    df["offset_arcsec"] = target.separation(sources).to(u.arcsec).value

    # r-band AB magnitude from nanomaggies, then Galactic-extinction corrected
    # mw_transmission_r is the fraction of flux transmitted; dividing removes dust attenuation
    df["mag_r"] = _mag_from_nanomaggy(df["flux_r"].values)
    trans = pd.to_numeric(df["mw_transmission_r"], errors="coerce").values
    with np.errstate(invalid="ignore", divide="ignore"):
        ext_corr = np.where((trans > 0) & np.isfinite(trans), -2.5 * np.log10(trans), np.nan)
    df["mag_r_corr"] = df["mag_r"] - ext_corr   # subtract extinction (brighter after correction)

    # Chance alignment probability uses extinction-corrected magnitude
    loc_error_arcsec = loc_error.to(u.arcsec).value
    p_cc = chance_alignment_prob(df["offset_arcsec"].values, df["mag_r_corr"].values, loc_error_arcsec)
    df["P_cc"] = np.where(df["mag_r_corr"].isna(), "no_mag_r", p_cc.round(6).astype(str))

    return df



# ---------------------------------------------------------------------------
# Aladin Lite notebook  (opened via Jupyter Lab)
# ---------------------------------------------------------------------------

def open_aladin(
    ra: float,
    dec: float,
    loc_error: u.Quantity,
    df_ls: pd.DataFrame,
) -> None:
    """Generate a temporary Jupyter notebook with an ipyaladin widget and open it."""
    tmp_dir = Path(tempfile.mkdtemp())

    fov_deg   = loc_error.to(u.deg).value * 8
    bg_survey = (
        "CDS/P/DESI-Legacy-Surveys/DR10/color"
        if not df_ls.empty
        else "CDS/P/2MASS/color"
    )

    def _cell(src: str, cell_type: str = "code") -> dict:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": src,
            "outputs": [],
            "execution_count": None,
        }
        if cell_type != "code":
            del cell["outputs"]
            del cell["execution_count"]
        return cell

    radius_deg = loc_error.to(u.deg).value
    has_ls     = not df_ls.empty

    # ── shared NED cell (always commented) ───────────────────────────────────
    ned_cell = _cell(
        "# ── NED cross-match ──────────────────────────────────────────────────\n"
        "# NOTE: use only for small regions (< ~10 arcmin).\n"
        "# NED may fail or time out for larger regions.\n"
        "#\n"
        "# from astroquery.ipac.ned import Ned\n"
        "#\n"
        "# coord = SkyCoord(ra, dec, unit='deg', frame='icrs')\n"
        "# result = Ned.query_region(coord, radius=loc_error_deg * u.deg)\n"
        "# df_ned = result.to_pandas()\n"
        "#\n"
        "# # Angular offset\n"
        "# sources = SkyCoord(df_ned['RA'].values, df_ned['DEC'].values, unit='deg')\n"
        "# df_ned['offset_arcsec'] = SkyCoord(ra, dec, unit='deg').separation(sources).arcsec\n"
        "#\n"
        "# # Overlay in Aladin (purple)\n"
        "# t_ned = Table.from_pandas(df_ned[['RA', 'DEC', 'Object Name', 'Redshift', 'offset_arcsec']])\n"
        "# t_ned.rename_column('RA', 'ra'); t_ned.rename_column('DEC', 'dec')\n"
        "# aladin.add_table(t_ned, color='purple', shape='circle')\n"
        "#\n"
        "# display(df_ned[['Object Name', 'RA', 'DEC', 'Type', 'Redshift', 'offset_arcsec']])\n"
    )

    if has_ls:
        # ── CASE 1: Legacy Survey coverage ───────────────────────────────────
        cells = [
            _cell(
                "from ipyaladin import Aladin\n"
                "from astropy.coordinates import SkyCoord\n"
                "from astropy.table import Table\n"
                "import astropy.units as u\n"
                "from regions import CircleSkyRegion\n"
                "from IPython.display import display\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import dl.queryClient as qc\n"
                "\n"
                "# ── parameters ───────────────────────────────────────────────\n"
                f"ra            = {ra}\n"
                f"dec           = {dec}\n"
                f"loc_error_deg = {radius_deg}\n"
                "\n"
                "# ── DataLab query (Legacy Survey DR10) ───────────────────────\n"
                f"query = \"\"\"\n"
                f"WITH cone AS (\n"
                f"    SELECT ls_id, ra, dec, type, flux_r, mw_transmission_r\n"
                f"    FROM ls_dr10.tractor\n"
                f"    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})\n"
                f"      AND type != 'PSF'\n"
                f"      AND maskbits = 0\n"
                f")\n"
                f"SELECT c.ls_id, c.ra, c.dec, c.type, c.flux_r, c.mw_transmission_r,\n"
                f"       p.z_phot_mean, p.z_spec, p.z_phot_std\n"
                f"FROM cone c\n"
                f"JOIN ls_dr10.photo_z p ON c.ls_id = p.ls_id\n"
                f"WHERE p.z_phot_mean IS NOT NULL;\n"
                f"\"\"\"\n"
                "df_ls = qc.query(sql=query, fmt='pandas')\n"
                "\n"
                "# Angular offset from transient\n"
                "target  = SkyCoord(ra, dec, unit='deg', frame='icrs')\n"
                "sources = SkyCoord(df_ls['ra'].values, df_ls['dec'].values, unit='deg', frame='icrs')\n"
                "df_ls['offset_arcsec'] = target.separation(sources).arcsec\n"
                "\n"
                "# r-band AB magnitude from nanomaggies, corrected for Galactic extinction\n"
                "# mw_transmission_r = fraction of flux transmitted (from SFD dust map)\n"
                "flux = df_ls['flux_r'].values.astype(float)\n"
                "df_ls['mag_r'] = np.where(flux > 0, 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan)), np.nan)\n"
                "trans = df_ls['mw_transmission_r'].values.astype(float)\n"
                "ext_corr = np.where((trans > 0) & np.isfinite(trans), -2.5 * np.log10(np.where(trans > 0, trans, np.nan)), np.nan)\n"
                "df_ls['mag_r_corr'] = df_ls['mag_r'] - ext_corr\n"
                "\n"
                "# Chance alignment probability (Bloom et al. 2002) — uses extinction-corrected magnitude\n"
                "# r = err if err > offset else sqrt(offset^2 + err^2)\n"
                "# sigma(<m) = 10^(0.334*(m-22.963)+4.32) / (3600^2 * 0.334 * ln10)\n"
                "# P_cc = 1 - exp(-pi * r^2 * sigma)\n"
                f"err_arcsec = {radius_deg * 3600.0}\n"
                "offset = df_ls['offset_arcsec'].values\n"
                "r = np.where(err_arcsec > offset, err_arcsec, np.sqrt(offset**2 + err_arcsec**2))\n"
                "norm = (3600.0**2) * 0.334 * np.log(10.0)\n"
                "sigma = 10.0 ** (0.334 * (df_ls['mag_r_corr'] - 22.963) + 4.32) / norm\n"
                "p_cc_vals = 1.0 - np.exp(-np.pi * r**2 * sigma)\n"
                "df_ls['P_cc'] = np.where(df_ls['mag_r_corr'].isna(), 'no_mag_r', p_cc_vals.round(6).astype(str))\n"
                "\n"
                "print(f'{len(df_ls)} extended sources found in Legacy Survey DR10')\n"
                "\n"
                "# ── Aladin widget (Legacy Survey DR10) ───────────────────────\n"
                f"aladin = Aladin(target=f'{{ra}} {{dec}}', fov={fov_deg},\n"
                "                survey='CDS/P/DESI-Legacy-Surveys/DR10/color')\n"
                "display(aladin)\n"
                "\n"
                "center = SkyCoord(ra, dec, unit='deg', frame='icrs')\n"
                "region = CircleSkyRegion(center=center, radius=loc_error_deg * u.deg)\n"
                "aladin.add_graphic_overlay_from_region([region], color='yellow', linewidth=2)\n"
                "\n"
                "t_ls = Table.from_pandas(df_ls[['ra', 'dec', 'type', 'z_phot_mean', 'mag_r', 'mag_r_corr', 'offset_arcsec', 'P_cc']])\n"
                "aladin.add_table(t_ls, color='red', shape='circle')\n"
                "\n"
                "# ── Table sorted by P_cc ────────────────────────────────────\n"
                "df_display = df_ls.copy()\n"
                "for col in ['z_phot_mean', 'mag_r', 'mag_r_corr', 'offset_arcsec']:\n"
                "    df_display[col] = df_display[col].apply(lambda x: f'{x:.4g}' if pd.notna(x) else x)\n"
                "df_display['P_cc'] = pd.to_numeric(df_display['P_cc'], errors='coerce').apply(\n"
                "    lambda x: f'{x:.4g}' if pd.notna(x) else 'no_mag_r'\n"
                ")\n"
                "df_display['_sort'] = pd.to_numeric(df_display['P_cc'], errors='coerce')\n"
                "display(df_display.sort_values('_sort').drop(columns='_sort').reset_index(drop=True))\n"
            ),
            ned_cell,
        ]

    else:
        # ── CASE 2: no Legacy Survey coverage ────────────────────────────────
        cells = [
            _cell(
                "from ipyaladin import Aladin\n"
                "from astropy.coordinates import SkyCoord\n"
                "from astropy.table import Table\n"
                "from astroquery.vizier import Vizier\n"
                "import astropy.units as u\n"
                "from regions import CircleSkyRegion\n"
                "from IPython.display import display\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "\n"
                "# ── parameters ───────────────────────────────────────────────\n"
                f"ra            = {ra}\n"
                f"dec           = {dec}\n"
                f"loc_error_deg = {radius_deg}\n"
                "\n"
                "# ── Survey selection — uncomment ONE ────────────────────────\n"
                "survey = 'CDS/P/2MASS/color'                       # 2MASS near-IR  ← default\n"
                "# survey = 'CDS/P/2MASS/K'                         # 2MASS K-band\n"
                "# survey = 'CDS/P/allWISE/color'                   # WISE mid-IR\n"
                "# survey = 'CDS/P/WISE/W1'                         # WISE W1\n"
                "# survey = 'CDS/P/PanSTARRS/DR1/color-i-r-g'      # Pan-STARRS optical\n"
                "# survey = 'CDS/P/PanSTARRS/DR1/color-z-zg-g'     # Pan-STARRS z-band\n"
                "# survey = 'CDS/P/DESI-Legacy-Surveys/DR10/color'  # Legacy Survey DR10\n"
                "\n"
                "# ── Aladin widget ─────────────────────────────────────────────\n"
                f"aladin = Aladin(target=f'{{ra}} {{dec}}', fov={fov_deg}, survey=survey)\n"
                "display(aladin)\n"
                "\n"
                "center = SkyCoord(ra, dec, unit='deg', frame='icrs')\n"
                "region = CircleSkyRegion(center=center, radius=loc_error_deg * u.deg)\n"
                "aladin.add_graphic_overlay_from_region([region], color='yellow', linewidth=2)\n"
                "\n"
                "# ── Galactic extinction at transient position (SFD98 via IRSA) ──\n"
                "# R_B = 4.07 (Schlafly & Finkbeiner 2011)\n"
                "try:\n"
                "    from astroquery.irsa_dust import IrsaDust\n"
                "    dust_table = IrsaDust.get_extinction_table(center)\n"
                "    ebv = float(dust_table['E(B-V)'][0])   # SFD E(B-V)\n"
                "    A_r = 2.285 * ebv\n"
                "    print(f'E(B-V) = {ebv:.4f},  A_r = {A_r:.4f} mag')\n"
                "except Exception as e:\n"
                "    A_r = 0.0\n"
                "    print(f'[WARNING] Could not get dust extinction: {e}. Using A_B = 0.')\n"
                "\n"
                "# ── GLADE+ (VII/291) — all-sky galaxy catalog ───────────────\n"
                "# B → r empirical conversion from 50-region GLADE × Legacy Survey DR10 cross-match:\n"
                "# B = 1.0645·r + 0.0366  →  r = (B − 0.0366) / 1.0645  (both magnitudes uncorrected)\n"
                "# r-band Galactic extinction applied after conversion (Schlafly & Finkbeiner 2011: R_r = 2.285)\n"
                "B2R_SLOPE     = 1.0645\n"
                "B2R_INTERCEPT = 0.0366\n"
                "try:\n"
                "    v = Vizier(columns=['RAJ2000','DEJ2000','Dist','z','Bmag','Kmag','W1mag','Type'],\n"
                "               row_limit=-1)\n"
                "    res = v.query_region(center, radius=loc_error_deg*u.deg, catalog='VII/291')\n"
                "    if res:\n"
                "        df = res[0].to_pandas().rename(columns={'RAJ2000':'ra','DEJ2000':'dec'})\n"
                "        sources = SkyCoord(df['ra'].values.astype(float),\n"
                "                           df['dec'].values.astype(float), unit='deg')\n"
                "        df['offset_arcsec'] = center.separation(sources).arcsec\n"
                "        mag_r_raw  = (df['Bmag'].values.astype(float) - B2R_INTERCEPT) / B2R_SLOPE\n"
                "        mag_r_corr = mag_r_raw - A_r\n"
                "        df['mag_r_est']  = mag_r_raw\n"
                "        df['mag_r_corr'] = mag_r_corr\n"
                f"        err_arcsec = {radius_deg * 3600.0}\n"
                "        off  = df['offset_arcsec'].values\n"
                "        r    = np.where(err_arcsec > off, err_arcsec, np.sqrt(off**2 + err_arcsec**2))\n"
                "        norm = (3600.0**2) * 0.334 * np.log(10.0)\n"
                "        sig  = 10.0 ** (0.334 * (mag_r_corr - 22.963) + 4.32) / norm\n"
                "        p_cc = 1 - np.exp(-np.pi * r**2 * sig)\n"
                "        df['P_cc'] = np.where(np.isnan(mag_r_corr), 'no_mag', p_cc.round(6).astype(str))\n"
                "        for col in ['Bmag','mag_r_est','mag_r_corr','Kmag','W1mag','z','Dist','offset_arcsec']:\n"
                "            if col in df:\n"
                "                df[col] = df[col].apply(lambda x: f'{x:.4g}' if pd.notna(x) and x!='' else x)\n"
                "        keep = [c for c in ['ra','dec','Type','Bmag','mag_r_est','mag_r_corr','Kmag','z','Dist','offset_arcsec','P_cc'] if c in df]\n"
                "        aladin.add_table(Table.from_pandas(df[keep].fillna('')), color='lime', shape='circle')\n"
                "        df['_sort'] = pd.to_numeric(df['P_cc'], errors='coerce')\n"
                "        print(f'{len(df)} galaxies in GLADE+ (green in Aladin)')\n"
                "        display(df[keep].assign(_sort=df['_sort']).sort_values('_sort')\n"
                "                        .drop(columns='_sort').reset_index(drop=True))\n"
                "    else:\n"
                "        print('No results in GLADE+ for this region.')\n"
                "except Exception as e:\n"
                "    print(f'Error GLADE+: {e}')\n"
                "\n"
                "# ── NED — NOTE: only for regions < ~10 arcmin ───────────────\n"
                "try:\n"
                "    from astroquery.ipac.ned import Ned\n"
                "    Ned.TIMEOUT = 120\n"
                "    result = Ned.query_region(center, radius=loc_error_deg * u.deg)\n"
                "    df_ned = result.to_pandas()\n"
                "    if not df_ned.empty:\n"
                "        sources = SkyCoord(df_ned['RA'].values, df_ned['DEC'].values, unit='deg')\n"
                "        df_ned['offset_arcsec'] = center.separation(sources).arcsec\n"
                "        df_ned['offset_arcsec'] = df_ned['offset_arcsec'].apply(lambda x: f'{x:.4g}')\n"
                "        t_ned = Table.from_pandas(df_ned[['RA','DEC','Object Name','Type','Redshift','offset_arcsec']].fillna(''))\n"
                "        t_ned.rename_column('RA','ra'); t_ned.rename_column('DEC','dec')\n"
                "        aladin.add_table(t_ned, color='purple', shape='circle')\n"
                "        print(f'{len(df_ned)} NED objects (purple in Aladin)')\n"
                "        display(df_ned[['Object Name','RA','DEC','Type','Redshift','offset_arcsec']]\n"
                "                      .sort_values('offset_arcsec').reset_index(drop=True))\n"
                "    else:\n"
                "        print('No NED results.')\n"
                "except Exception as e:\n"
                "    print(f'Error NED: {e}')\n"
            ),
            _cell(
                "# ── Chance Alignment Probability (Bloom et al. 2002) ─────────\n"
                "# P_cc = 1 - exp(-pi * r^2 * sigma(<mag_r))\n"
                "# r = err if err > offset else sqrt(offset^2 + err^2)\n"
                "# sigma(<m) = 10^(0.334*(m-22.963)+4.32) / (3600^2 * 0.334 * ln10)\n"
                "\n"
                "def chance_alignment_prob(ra_gal, dec_gal, mag_r,\n"
                "                          ra_transient=ra, dec_transient=dec,\n"
                f"                          loc_error_arcsec={radius_deg * 3600.0}):\n"
                "    \"\"\"\n"
                "    Bloom et al. (2002) chance alignment probability.\n"
                "    Parameters:\n"
                "        ra_gal, dec_gal  : candidate galaxy coordinates (deg)\n"
                "        mag_r            : galaxy magnitude (r-band AB)\n"
                "        loc_error_arcsec : localization radius (arcsec)\n"
                "    Returns P_cc in [0, 1]  (0 = likely host, 1 = chance coincidence)\n"
                "    \"\"\"\n"
                "    import math\n"
                "    target = SkyCoord(ra_transient, dec_transient, unit='deg', frame='icrs')\n"
                "    source = SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs')\n"
                "    offset_arcsec = target.separation(source).arcsec\n"
                "    r     = loc_error_arcsec if loc_error_arcsec > offset_arcsec else math.sqrt(offset_arcsec**2 + loc_error_arcsec**2)\n"
                "    norm  = (3600.0**2) * 0.334 * math.log(10.0)\n"
                "    sigma = 10.0 ** (0.334 * (mag_r - 22.963) + 4.32) / norm\n"
                "    p_cc  = 1.0 - math.exp(-math.pi * r**2 * sigma)\n"
                "    print(f'offset = {offset_arcsec:.4g} arcsec')\n"
                "    print(f'r_eff  = {r:.4g} arcsec')\n"
                "    print(f'P_cc   = {p_cc:.4g}')\n"
                "    return p_cc\n"
                "\n"
                "# Example:\n"
                f"# chance_alignment_prob(ra_gal={ra:.4f}, dec_gal={dec:.4f}, mag_r=20.5)\n"
            ),
        ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }

    nb_path = tmp_dir / "aladin_view.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1))
    print(f"\n    Opening Jupyter Lab: {nb_path}")
    print("    (Ctrl+C to close)")
    subprocess.run(["jupyter", "lab", str(nb_path)])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args      = parse_args()
    ra        = args.ra
    dec       = args.dec
    loc_error = parse_error(args.error)

    print(f"\nTarget: RA={ra:.4f}  Dec={dec:.4f}  radius={loc_error:.3f}")
    print("=" * 60)

    # ── 1. Legacy Survey DR10 ────────────────────────────────────────────────
    print("\n[1] Legacy Survey DR10 (NOIRLab DataLab)...")
    df_ls = query_legacy(ra, dec, loc_error)

    if not df_ls.empty:
        print(f"    {len(df_ls)} extended sources found.\n")
        display_cols = [c for c in ["ls_id", "ra", "dec", "type",
                                    "z_phot_mean", "mag_r", "mag_r_corr", "offset_arcsec", "P_cc"]
                        if c in df_ls.columns]
        df_print = df_ls[display_cols].copy()
        for col in ["z_phot_mean", "mag_r", "mag_r_corr", "offset_arcsec"]:
            if col in df_print.columns:
                df_print[col] = df_print[col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else x)
        df_print["P_cc"] = pd.to_numeric(df_print["P_cc"], errors="coerce").apply(
            lambda x: f"{x:.4g}" if pd.notna(x) else "no_mag_r"
        )
        df_print["_sort"] = pd.to_numeric(df_print["P_cc"], errors="coerce")
        print(df_print.sort_values("_sort").drop(columns="_sort").to_string(index=False))
    else:
        print("    No Legacy Survey coverage for this position.")

    # ── 2. Open Aladin ────────────────────────────────────────────────────────
    open_aladin(ra, dec, loc_error, df_ls)


if __name__ == "__main__":
    main()
