#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatial correspondence between the normative basal-forebrain cortical
expression pattern and molecular PET maps.

"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot
from neuromaps import datasets, transforms
from scipy import stats
from scipy.io import loadmat


N_PARCELS = 400
N_PARCELS_HEMI = 200
DEFAULT_FSAVERAGE_DENSITY = "10k"  # fsaverage5


PET_MAPS = {
    "VAChT": [
        {"source": "aghourian2017", "desc": "feobv", "space": "MNI152", "res": "1mm"},
        {"source": "bedard2019", "desc": "feobv", "space": "MNI152", "res": "1mm"},
        {"source": "tuominen", "desc": "feobv", "space": "MNI152", "res": "2mm"},
    ],
    "NET": [
        {"source": "ding2010", "desc": "mrb", "space": "MNI152", "res": "1mm"},
    ],
    "DAT": [
        {"source": "sasaki2012", "desc": "fepe2i", "space": "MNI152", "res": "1mm"},
    ],
    "5HTT": [
        {"source": "beliveau2017", "desc": "dasb", "space": "MNI152", "res": "1mm"},
    ],
}

# Set paths

G1_TEMPLATE = Path("/path/to/controlTemp_G1.xlsx")
LH_ANNOT = Path("/path/to/lh.Schaefer2018_400Parcels_7Networks_order.annot")
RH_ANNOT = Path("/path/to/rh.Schaefer2018_400Parcels_7Networks_order.annot")
SPINS_FILE = Path("/path/to/spins_400x10000.mat")
OUT_DIR = Path("/path/to/output/pet_correspondence")

# fsaverage5 corresponds to the 10k fsaverage density in neuromaps.
FSAVERAGE_DENSITY = "10k"

# Molecular systems to analyse.
SYSTEMS = ["VAChT", "NET", "DAT", "5HTT"]


def load_vector(path: str | Path) -> np.ndarray:
    """Load a one-dimensional numeric vector from xlsx/csv/txt/npy."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        x = np.load(path)
    elif suffix in {".xlsx", ".xls"}:
        x = pd.read_excel(path, header=None).iloc[:, 0].to_numpy()
    elif suffix == ".csv":
        x = pd.read_csv(path, header=None).iloc[:, 0].to_numpy()
    elif suffix in {".txt", ".tsv"}:
        x = pd.read_csv(path, sep=None, engine="python", header=None).iloc[:, 0].to_numpy()
    else:
        raise ValueError(f"Unsupported vector format: {path}")

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != N_PARCELS:
        raise ValueError(f"Expected {N_PARCELS} values in {path}; found {x.size}.")
    return x


def zscore_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    out = np.full(x.shape, np.nan, dtype=float)
    ok = np.isfinite(x)

    if ok.sum() < 5:
        return out

    sd = np.nanstd(x[ok])
    if sd == 0:
        raise ValueError("Cannot z-standardize a constant vector.")

    out[ok] = (x[ok] - np.nanmean(x[ok])) / sd
    return out


def load_spins(path: str | Path) -> np.ndarray:
    """Load a zero-based parcel x permutation spin matrix."""
    mat = loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]

    spins = None
    for key in ["spins", "spin", "spins_400", "perm_id", "spin_idx"]:
        if key in mat:
            spins = mat[key]
            break

    if spins is None:
        arrays = [mat[k] for k in keys if isinstance(mat[k], np.ndarray) and mat[k].ndim == 2]
        if len(arrays) == 1:
            spins = arrays[0]
        else:
            raise ValueError(
                f"Could not uniquely identify a 2D spin matrix in {path}. Variables: {keys}"
            )

    spins = np.asarray(spins, dtype=int)

    if spins.shape[0] != N_PARCELS and spins.shape[1] == N_PARCELS:
        spins = spins.T

    if spins.shape[0] != N_PARCELS:
        raise ValueError(f"Expected spin matrix with 400 rows; got {spins.shape}.")

    # Accept legacy MATLAB-style 1-based matrices, but use zero-based internally.
    if spins.min() == 1 and spins.max() == N_PARCELS:
        spins = spins - 1

    if spins.min() < 0 or spins.max() >= N_PARCELS:
        raise ValueError(
            f"Spin indices must be in [0, {N_PARCELS - 1}]; "
            f"observed [{spins.min()}, {spins.max()}]."
        )

    print(f"Loaded spins: shape={spins.shape}, range=[{spins.min()}, {spins.max()}]")
    return spins


def spin_corr(
    x: np.ndarray,
    y: np.ndarray,
    spins: np.ndarray,
    method: str = "spearman",
) -> tuple[float, float, np.ndarray]:
    """Two-sided spatial spin test using rotations of y."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size != N_PARCELS or y.size != N_PARCELS:
        raise ValueError("spin_corr expects two 400-parcel vectors.")

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan, np.nan, np.full(spins.shape[1], np.nan)

    corr = stats.spearmanr if method == "spearman" else stats.pearsonr
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'.")

    r_obs = float(corr(x[ok], y[ok])[0])
    nulls = np.full(spins.shape[1], np.nan, dtype=float)

    for i in range(spins.shape[1]):
        y_perm = y[spins[:, i]]
        ok_perm = np.isfinite(x) & np.isfinite(y_perm)
        if ok_perm.sum() >= 10:
            nulls[i] = float(corr(x[ok_perm], y_perm[ok_perm])[0])

    finite = np.isfinite(nulls)
    p_spin = (np.sum(np.abs(nulls[finite]) >= abs(r_obs)) + 1) / (finite.sum() + 1)
    return r_obs, float(p_spin), nulls


def fetch_pet(spec: dict):
    print(f"Fetching: {spec}")
    return datasets.fetch_annotation(**spec)


def _first_annotation_object(obj):
    """Unwrap common single-annotation containers returned by neuromaps."""
    if isinstance(obj, dict):
        vals = list(obj.values())
        if len(vals) == 0:
            raise ValueError("neuromaps returned an empty annotation dictionary.")
        return vals[0]
    return obj


def transform_to_fsaverage(fetched, spec: dict, density: str):
    space = spec["space"].lower()

    if space == "fsaverage":
        return fetched

    if space == "mni152":
        fetched = _first_annotation_object(fetched)
        return transforms.mni152_to_fsaverage(fetched, fsavg_density=density)

    raise ValueError(f"Unsupported annotation space: {spec['space']}")


def get_gifti_data(img) -> np.ndarray:
    data = img.agg_data()
    if isinstance(data, tuple):
        data = np.column_stack([np.asarray(d).reshape(-1) for d in data])[:, 0]
    data = np.asarray(data).squeeze()
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(float)


def load_surface_pair(surf) -> tuple[np.ndarray, np.ndarray]:
    """Return LH/RH surface arrays from common neuromaps output types."""
    if isinstance(surf, dict):
        vals = list(surf.values())
        candidates = [
            v for v in vals
            if hasattr(v, "agg_data") or isinstance(v, (str, bytes, Path))
        ]
        if len(candidates) < 2:
            raise ValueError(f"Could not identify LH/RH surfaces from keys: {list(surf.keys())}")
        lh_obj, rh_obj = candidates[:2]
    elif isinstance(surf, (tuple, list)) and len(surf) == 2:
        lh_obj, rh_obj = surf
    else:
        raise ValueError(f"Could not interpret neuromaps surface output: {type(surf)}")

    def load_one(obj):
        img = obj if hasattr(obj, "agg_data") else nib.load(str(obj))
        return get_gifti_data(img)

    return load_one(lh_obj), load_one(rh_obj)


def valid_parcel_ids(labels: np.ndarray, names, expected_n: int) -> list[int]:
    """Return ordered cortical parcel IDs, excluding unknown / medial wall labels."""
    ids = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    clean = []

    for pid in ids:
        name = ""
        if pid < len(names):
            raw = names[pid]
            name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if "unknown" in name.lower() or "medial" in name.lower():
            continue
        clean.append(pid)

    if len(clean) != expected_n:
        raise ValueError(
            f"Expected {expected_n} cortical parcels in annotation; found {len(clean)}. "
            "Check that the annotation matches the requested fsaverage density."
        )
    return clean


def parcellate_surface_data(
    data: np.ndarray,
    labels: np.ndarray,
    names,
    expected_n: int = N_PARCELS_HEMI,
) -> np.ndarray:
    data = np.asarray(data, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if data.size != labels.size:
        raise ValueError(
            f"Surface data has {data.size} vertices but annotation has {labels.size}. "
            "Use Schaefer annotations at the same fsaverage density as the transformed PET map."
        )

    parcel_ids = valid_parcel_ids(labels, names, expected_n)
    vals = np.full(expected_n, np.nan, dtype=float)

    for i, pid in enumerate(parcel_ids):
        mask = labels == pid
        if np.any(mask):
            vals[i] = np.nanmean(data[mask])

    return vals


def parcellate_pet(
    fetched,
    spec: dict,
    density: str,
    lh_labels: np.ndarray,
    lh_names,
    rh_labels: np.ndarray,
    rh_names,
) -> np.ndarray:
    surf = transform_to_fsaverage(fetched, spec, density)
    lh_data, rh_data = load_surface_pair(surf)

    lh_parc = parcellate_surface_data(lh_data, lh_labels, lh_names)
    rh_parc = parcellate_surface_data(rh_data, rh_labels, rh_names)
    parc = np.concatenate([lh_parc, rh_parc])

    if parc.size != N_PARCELS:
        raise RuntimeError(f"Expected 400 parcels after concatenation; got {parc.size}.")
    return parc


def spec_label(spec: dict) -> str:
    bits = [spec["source"], spec["desc"], spec["space"]]
    if "res" in spec:
        bits.append(spec["res"])
    if "den" in spec:
        bits.append(spec["den"])
    return "_".join(bits)


def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    g1 = load_vector(G1_TEMPLATE)
    g1_z = zscore_vec(g1)
    spins = load_spins(SPINS_FILE)

    print("Loading Schaefer annotations...")
    lh_labels, _, lh_names = read_annot(str(LH_ANNOT))
    rh_labels, _, rh_names = read_annot(str(RH_ANNOT))
    print(f"LH vertices: {len(lh_labels)}")
    print(f"RH vertices: {len(rh_labels)}")

    individual_rows = []
    pet_vectors: dict[str, list[dict]] = {}

    for system in SYSTEMS:
        specs = PET_MAPS[system]
        pet_vectors[system] = []

        print("\n" + "=" * 80)
        print(system)
        print("=" * 80)

        for spec in specs:
            label = spec_label(spec)
            print(f"\nFetching/parcellating {label}")

            try:
                fetched = fetch_pet(spec)
                parc = parcellate_pet(
                    fetched,
                    spec,
                    FSAVERAGE_DENSITY,
                    lh_labels,
                    lh_names,
                    rh_labels,
                    rh_names,
                )
                parc_z = zscore_vec(parc)

                rho, p_spin, _ = spin_corr(g1_z, parc_z, spins, method="spearman")
                r, p_spin_r, _ = spin_corr(g1_z, parc_z, spins, method="pearson")

                pet_vectors[system].append(
                    {"label": label, "spec": spec, "vector": parc_z}
                )

                print(f"  Spearman rho={rho:+.3f}, spin p={p_spin:.4f}")
                print(f"  Pearson  r  ={r:+.3f}, spin p={p_spin_r:.4f}")

                pd.DataFrame(
                    {
                        "parcel": np.arange(1, N_PARCELS + 1),
                        "G1_template_z": g1_z,
                        f"{label}_z": parc_z,
                    }
                ).to_csv(out_dir / f"parcellated_{label}.csv", index=False)

                individual_rows.append(
                    {
                        "system": system,
                        "map": label,
                        "source": spec["source"],
                        "desc": spec["desc"],
                        "space": spec["space"],
                        "res_or_den": spec.get("res", spec.get("den", "")),
                        "spearman_rho": rho,
                        "spearman_spin_p": p_spin,
                        "pearson_r": r,
                        "pearson_spin_p": p_spin_r,
                        "n_valid": int(np.sum(np.isfinite(g1_z) & np.isfinite(parc_z))),
                        "error": "",
                    }
                )

            except Exception as exc:
                print(f"  FAILED {label}: {exc}")
                individual_rows.append(
                    {
                        "system": system,
                        "map": label,
                        "source": spec["source"],
                        "desc": spec["desc"],
                        "space": spec["space"],
                        "res_or_den": spec.get("res", spec.get("den", "")),
                        "spearman_rho": np.nan,
                        "spearman_spin_p": np.nan,
                        "pearson_r": np.nan,
                        "pearson_spin_p": np.nan,
                        "n_valid": np.nan,
                        "error": str(exc),
                    }
                )

    individual_df = pd.DataFrame(individual_rows)
    individual_df.to_csv(out_dir / "individual_PET_G1_correlations.csv", index=False)

    consensus_rows = []
    consensus_matrix = pd.DataFrame(
        {
            "parcel": np.arange(1, N_PARCELS + 1),
            "G1_template_z": g1_z,
        }
    )

    for system, items in pet_vectors.items():
        if len(items) == 0:
            print(f"\nNo successful maps for {system}; skipping consensus.")
            continue

        print("\n" + "=" * 80)
        print(f"Consensus {system}")
        print("=" * 80)

        matrix = np.column_stack([item["vector"] for item in items])
        consensus = zscore_vec(np.nanmean(matrix, axis=1))

        rho, p_spin, _ = spin_corr(g1_z, consensus, spins, method="spearman")
        r, p_spin_r, _ = spin_corr(g1_z, consensus, spins, method="pearson")

        print(f"  n_maps={len(items)}")
        print(f"  Spearman rho={rho:+.3f}, spin p={p_spin:.4f}")
        print(f"  Pearson  r  ={r:+.3f}, spin p={p_spin_r:.4f}")

        consensus_matrix[f"{system}_consensus_z"] = consensus
        pd.DataFrame(
            {
                "parcel": np.arange(1, N_PARCELS + 1),
                f"{system}_consensus_z": consensus,
            }
        ).to_csv(out_dir / f"consensus_{system}_Schaefer400.csv", index=False)

        consensus_rows.append(
            {
                "system": system,
                "n_maps": len(items),
                "maps_included": "; ".join(item["label"] for item in items),
                "spearman_rho": rho,
                "spearman_spin_p": p_spin,
                "pearson_r": r,
                "pearson_spin_p": p_spin_r,
            }
        )

    pd.DataFrame(consensus_rows).to_csv(
        out_dir / "consensus_PET_G1_correlations.csv", index=False
    )
    consensus_matrix.to_csv(
        out_dir / "G1_and_consensus_PET_Schaefer400.csv", index=False
    )

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
