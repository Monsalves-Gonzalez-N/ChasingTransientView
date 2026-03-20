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


def chance_alignment_prob(offset_arcsec: np.ndarray, mag_r: np.ndarray) -> np.ndarray:
    """
    Bloom et al. (2002) chance coincidence probability.

        P_cc = 1 - exp(-pi * offset^2 * sigma(< m_r))

    where sigma(<m_r) [per arcsec^2] is the cumulative surface density of
    galaxies brighter than m_r (r-band AB), approximated as:

        sigma(<m_r) = (1/3600) * 10^(0.33 * (m_r - 24))

    Parameters
    ----------
    offset_arcsec : angular separation from the transient (arcsec)
    mag_r         : r-band AB magnitude of the galaxy

    Returns
    -------
    P_cc in [0, 1]  (NaN when magnitude is undefined)
    """
    offset = np.asarray(offset_arcsec, dtype=float)
    mag_r  = np.asarray(mag_r, dtype=float)
    sigma  = (1.0 / 3600.0) * 10.0 ** (0.33 * (mag_r - 24.0))   # per arcsec²
    p_cc   = 1.0 - np.exp(-np.pi * offset**2 * sigma)
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
    SELECT ls_id, ra, dec, type, flux_r
    FROM ls_dr10.tractor
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})
      AND type != 'PSF'
      AND maskbits = 0
)
SELECT c.ls_id, c.ra, c.dec, c.type, c.flux_r,
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

    # r-band AB magnitude from nanomaggies
    df["mag_r"] = _mag_from_nanomaggy(df["flux_r"].values)

    # Chance alignment probability — "no_mag_r" when flux_r is missing
    p_cc = chance_alignment_prob(df["offset_arcsec"].values, df["mag_r"].values)
    df["P_cc"] = np.where(df["mag_r"].isna(), "no_mag_r", p_cc.round(6).astype(str))

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
                f"    SELECT ls_id, ra, dec, type, flux_r\n"
                f"    FROM ls_dr10.tractor\n"
                f"    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})\n"
                f"      AND type != 'PSF'\n"
                f"      AND maskbits = 0\n"
                f")\n"
                f"SELECT c.ls_id, c.ra, c.dec, c.type, c.flux_r,\n"
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
                "# r-band AB magnitude from nanomaggies\n"
                "flux = df_ls['flux_r'].values.astype(float)\n"
                "df_ls['mag_r'] = np.where(flux > 0, 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan)), np.nan)\n"
                "\n"
                "# Chance alignment probability (Bloom et al. 2002)\n"
                "# P_cc = 1 - exp(-pi * offset^2 * sigma(<mag_r)),  sigma in arcsec^-2\n"
                "sigma = (1.0 / 3600.0) * 10.0 ** (0.33 * (df_ls['mag_r'] - 24.0))\n"
                "p_cc_vals = 1.0 - np.exp(-np.pi * df_ls['offset_arcsec'] ** 2 * sigma)\n"
                "df_ls['P_cc'] = np.where(df_ls['mag_r'].isna(), 'no_mag_r', p_cc_vals.round(6).astype(str))\n"
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
                "t_ls = Table.from_pandas(df_ls[['ra', 'dec', 'type', 'z_phot_mean', 'mag_r', 'offset_arcsec', 'P_cc']])\n"
                "aladin.add_table(t_ls, color='red', shape='circle')\n"
                "\n"
                "# ── Table sorted by P_cc ────────────────────────────────────\n"
                "df_display = df_ls.copy()\n"
                "for col in ['z_phot_mean', 'mag_r', 'offset_arcsec']:\n"
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
                "# ── GLADE+ (VII/291) — all-sky galaxy catalog ───────────────\n"
                "# P_cc computed with Bmag (approximation to r-band)\n"
                "try:\n"
                "    v = Vizier(columns=['RAJ2000','DEJ2000','Dist','z','Bmag','Kmag','W1mag','Type'],\n"
                "               row_limit=-1)\n"
                "    res = v.query_region(center, radius=loc_error_deg*u.deg, catalog='VII/291')\n"
                "    if res:\n"
                "        df = res[0].to_pandas().rename(columns={'RAJ2000':'ra','DEJ2000':'dec'})\n"
                "        sources = SkyCoord(df['ra'].values.astype(float),\n"
                "                           df['dec'].values.astype(float), unit='deg')\n"
                "        df['offset_arcsec'] = center.separation(sources).arcsec\n"
                "        mag  = df['Bmag'].values.astype(float)\n"
                "        sig  = (1/3600) * 10**(0.33*(mag - 24))\n"
                "        p_cc = 1 - np.exp(-np.pi * df['offset_arcsec']**2 * sig)\n"
                "        df['P_cc'] = np.where(np.isnan(mag), 'no_mag', p_cc.round(6).astype(str))\n"
                "        for col in ['Bmag','Kmag','W1mag','z','Dist','offset_arcsec']:\n"
                "            if col in df:\n"
                "                df[col] = df[col].apply(lambda x: f'{x:.4g}' if pd.notna(x) and x!='' else x)\n"
                "        keep = [c for c in ['ra','dec','Type','Bmag','Kmag','z','Dist','offset_arcsec','P_cc'] if c in df]\n"
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
                "# Use to manually compute P_cc given coordinates and magnitude\n"
                "# of any source seen in Aladin or in the GLADE+/NED table.\n"
                "#\n"
                "# P_cc = 1 - exp(-pi * offset^2 * sigma(<mag_r))\n"
                "# sigma(<mag_r) [per arcsec^2] = (1/3600) * 10^(0.33*(mag_r - 24))\n"
                "\n"
                "def chance_alignment_prob(ra_gal, dec_gal, mag_r,\n"
                "                          ra_transient=ra, dec_transient=dec):\n"
                "    \"\"\"\n"
                "    Bloom et al. (2002) chance alignment probability.\n"
                "    Parameters:\n"
                "        ra_gal, dec_gal : candidate galaxy coordinates (deg)\n"
                "        mag_r           : galaxy magnitude (ideally r-band AB)\n"
                "        ra_transient, dec_transient : transient coordinates\n"
                "    Returns P_cc in [0, 1]  (0 = likely host, 1 = chance coincidence)\n"
                "    \"\"\"\n"
                "    target = SkyCoord(ra_transient, dec_transient, unit='deg', frame='icrs')\n"
                "    source = SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs')\n"
                "    offset_arcsec = target.separation(source).arcsec\n"
                "    sigma = (1.0 / 3600.0) * 10.0 ** (0.33 * (mag_r - 24.0))\n"
                "    p_cc  = 1.0 - np.exp(-np.pi * offset_arcsec**2 * sigma)\n"
                "    print(f'offset = {offset_arcsec:.4g} arcsec')\n"
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
                                    "z_phot_mean", "mag_r", "offset_arcsec", "P_cc"]
                        if c in df_ls.columns]
        df_print = df_ls[display_cols].copy()
        for col in ["z_phot_mean", "mag_r", "offset_arcsec"]:
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
