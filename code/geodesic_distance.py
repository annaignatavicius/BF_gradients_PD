#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate cortical geodesic distance from a surface-projected basal-forebrain
seed to Schaefer-400 cortical parcels.

The calculation is performed separately in each hemisphere on the fsaverage5
pial surface. Cortical / medial-wall vertices are defined by the FreeSurfer
cortex label. The surface-projected basal-forebrain seed is added to the cortical
mesh and expanded by a small number of face rings to ensure mesh connectivity.
Minimum geodesic distance from the seed vertices is then calculated with gdist,
back-projected to the full surface, and averaged within Schaefer-400 parcels.

Run this script once for Ch123 and once for Ch4/Ch4p, supplying the corresponding
left- and right-hemisphere surface seed files.

"""

from __future__ import annotations

from pathlib import Path

import gdist
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot, read_geometry, read_label
from scipy.io import savemat


N_PARCELS_HEMI = 200
N_PARCELS = 400


# Set paths


# Run the script separately for Ch123 and Ch4/Ch4p by changing these values.
SEED_NAME = "Ch123"  # e.g. "Ch123" or "Ch4"

LH_SEED = Path("/path/to/lh.BF_Ch123.fsavg5.shape.gii")
RH_SEED = Path("/path/to/rh.BF_Ch123.fsavg5.shape.gii")

LH_PIAL = Path("/path/to/fsaverage5/surf/lh.pial")
RH_PIAL = Path("/path/to/fsaverage5/surf/rh.pial")

LH_CORTEX = Path("/path/to/fsaverage5/label/lh.cortex.label")
RH_CORTEX = Path("/path/to/fsaverage5/label/rh.cortex.label")

LH_ANNOT = Path("/path/to/lh.Schaefer2018_400Parcels_7Networks_order.annot")
RH_ANNOT = Path("/path/to/rh.Schaefer2018_400Parcels_7Networks_order.annot")

OUT_DIR = Path("/path/to/output/geodesic")

# Analysis settings
RINGS = 1
FILL_ITERS = 2


def expand_keep_by_faces(keep_mask: np.ndarray, faces: np.ndarray, rings: int = 1) -> np.ndarray:
    """Expand a vertex mask by including vertices from touching faces."""
    keep = keep_mask.copy()
    for _ in range(rings):
        touched_faces = np.any(keep[faces], axis=1)
        verts_to_add = np.unique(faces[touched_faces].ravel())
        new_keep = keep.copy()
        new_keep[verts_to_add] = True
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return keep


def build_submesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a triangular submesh and return old->new vertex indices."""
    n_vertices = vertices.shape[0]
    old_to_new = -np.ones(n_vertices, dtype=np.int32)
    keep_idx = np.flatnonzero(keep_mask)
    old_to_new[keep_idx] = np.arange(keep_idx.size, dtype=np.int32)

    face_keep = keep_mask[faces].all(axis=1)
    sub_faces = old_to_new[faces[face_keep]]
    sub_vertices = vertices[keep_mask]

    return (
        sub_vertices.astype(np.float64),
        sub_faces.astype(np.int32),
        old_to_new,
    )


def write_gifti_surface(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    gii = nib.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=vertices.astype(np.float32),
            intent="NIFTI_INTENT_POINTSET",
        )
    )
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=faces.astype(np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
        )
    )
    nib.save(gii, str(out_path))


def write_gifti_shape(data: np.ndarray, out_path: Path) -> None:
    gii = nib.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(data=np.asarray(data, dtype=np.float32))
    )
    nib.save(gii, str(out_path))


def build_neighbor_list(faces: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Create a vertex-neighbor lookup from triangular faces."""
    neighbors = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.asarray(sorted(v), dtype=np.int32) for v in neighbors]


def fill_zero_distances(
    data: np.ndarray,
    faces: np.ndarray,
    fill_mask: np.ndarray,
    max_iters: int = 2,
) -> np.ndarray:
    """Fill isolated zero-valued cortical vertices from nonzero neighbors."""
    out = np.asarray(data, dtype=float).copy()
    neighbors = build_neighbor_list(faces, len(out))

    for _ in range(max_iters):
        zero_idx = np.flatnonzero((out == 0) & fill_mask)
        if zero_idx.size == 0:
            break

        changed = 0
        previous = out.copy()
        for vertex in zero_idx:
            vals = previous[neighbors[vertex]]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                out[vertex] = vals.mean()
                changed += 1

        if changed == 0:
            break

    return out


def load_seed(path: str | Path, n_vertices: int) -> np.ndarray:
    """Load a binary/nonzero GIFTI seed vector."""
    img = nib.load(str(path))
    if len(img.darrays) == 0:
        raise ValueError(f"No data arrays found in seed: {path}")
    seed = np.asarray(img.darrays[0].data).reshape(-1)
    if seed.size != n_vertices:
        raise ValueError(
            f"Seed {path} contains {seed.size} vertices; surface contains {n_vertices}."
        )
    seed = np.isfinite(seed) & (seed != 0)
    if seed.sum() == 0:
        raise ValueError(f"Seed contains no nonzero vertices: {path}")
    return seed


def clean_parcel_ids(labels: np.ndarray, names, cortex_mask: np.ndarray) -> list[int]:
    """Return the 200 cortical Schaefer parcel IDs in annotation order."""
    ids = sorted(int(v) for v in np.unique(labels[cortex_mask]) if int(v) >= 0)
    clean = []
    for pid in ids:
        name = ""
        if pid < len(names):
            raw = names[pid]
            name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if "unknown" in name.lower() or "medial" in name.lower():
            continue
        clean.append(pid)

    if len(clean) != N_PARCELS_HEMI:
        raise ValueError(
            f"Expected {N_PARCELS_HEMI} Schaefer parcels in hemisphere; found {len(clean)}."
        )
    return clean


def process_hemisphere(
    hemi: str,
    pial_path: str | Path,
    cortex_path: str | Path,
    seed_path: str | Path,
    annot_path: str | Path,
    out_dir: Path,
    seed_name: str,
    rings: int,
    fill_iters: int,
) -> np.ndarray:
    print(f"\n--- {hemi} ---")

    vertices, faces = read_geometry(str(pial_path))
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    n_vertices = vertices.shape[0]

    cortex_idx = read_label(str(cortex_path)).astype(np.int64)
    cortex_mask = np.zeros(n_vertices, dtype=bool)
    cortex_mask[np.unique(cortex_idx)] = True

    seed = load_seed(seed_path, n_vertices)
    seed_idx = np.flatnonzero(seed)

    # Include cortical vertices and projected seed vertices, then expand by a
    # small face ring so the seed projection remains connected to the surface.
    keep_mask = cortex_mask | seed
    keep_mask = expand_keep_by_faces(keep_mask, faces, rings=rings)

    sub_vertices, sub_faces, old_to_new = build_submesh(vertices, faces, keep_mask)
    source_indices = old_to_new[seed_idx]
    source_indices = source_indices[source_indices >= 0]

    if source_indices.size == 0 or sub_faces.size == 0:
        raise RuntimeError(
            f"No connected source vertices for {hemi}. Check seed/surface alignment or increase --rings."
        )

    dist_sub = gdist.compute_gdist(
        sub_vertices,
        sub_faces,
        source_indices=source_indices.astype(np.int32),
    ).astype(np.float32)

    dist_full = np.zeros(n_vertices, dtype=np.float32)
    dist_full[np.flatnonzero(keep_mask)] = dist_sub

    # Keep only cortical distances for the cortical output map.
    raw = dist_full.copy()
    raw[~cortex_mask] = 0.0

    # Rare zero-valued cortical vertices created around the surface boundary
    # are filled from neighboring nonzero cortical distances. Seed vertices
    # themselves are not altered.
    fill_mask = cortex_mask & (~seed)
    filled = fill_zero_distances(raw, faces, fill_mask, max_iters=fill_iters)

    prefix = f"{hemi}.BF_{seed_name}.fsavg5"
    write_gifti_shape(raw, out_dir / f"{prefix}.geodesic_raw.shape.gii")
    write_gifti_shape(filled, out_dir / f"{prefix}.geodesic_filled.shape.gii")
    write_gifti_surface(
        sub_vertices,
        sub_faces,
        out_dir / f"{prefix}.analysis_mesh.surf.gii",
    )

    labels, _, names = read_annot(str(annot_path))
    if labels.size != n_vertices:
        raise ValueError(
            f"Annotation {annot_path} has {labels.size} vertices; pial surface has {n_vertices}."
        )

    parcel_ids = clean_parcel_ids(labels, names, cortex_mask)
    parcel_distances = np.full(N_PARCELS_HEMI, np.nan, dtype=np.float32)

    for i, parcel_id in enumerate(parcel_ids):
        parcel_mask = (labels == parcel_id) & cortex_mask
        if np.any(parcel_mask):
            parcel_distances[i] = np.nanmean(filled[parcel_mask])

    print(
        f"vertices={n_vertices}, kept={keep_mask.sum()}, "
        f"seed_vertices={seed.sum()}, mean_parcel_distance={np.nanmean(parcel_distances):.2f}"
    )

    return parcel_distances


def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    lh = process_hemisphere(
        "lh",
        LH_PIAL,
        LH_CORTEX,
        LH_SEED,
        LH_ANNOT,
        out_dir,
        SEED_NAME,
        RINGS,
        FILL_ITERS,
    )
    rh = process_hemisphere(
        "rh",
        RH_PIAL,
        RH_CORTEX,
        RH_SEED,
        RH_ANNOT,
        out_dir,
        SEED_NAME,
        RINGS,
        FILL_ITERS,
    )

    distances = np.concatenate([lh, rh]).astype(np.float32)
    if distances.size != N_PARCELS:
        raise RuntimeError(f"Expected 400 parcel distances; got {distances.size}.")

    mat_path = out_dir / f"geodesic_distance_parcels_{SEED_NAME}.mat"
    csv_path = out_dir / f"geodesic_distance_parcels_{SEED_NAME}.csv"

    savemat(mat_path, {"D_parcel": distances})
    pd.DataFrame(
        {
            "parcel": np.arange(1, N_PARCELS + 1),
            "geodesic_distance": distances,
        }
    ).to_csv(csv_path, index=False)

    print("\nSaved:")
    print(f"  {mat_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
