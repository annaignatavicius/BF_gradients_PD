#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate spin-permutation indices for the Schaefer-400 parcellation.

Uses the Alexander-Bloch spherical rotation method implemented in neuromaps.
The Schaefer-400 parcellation contains 200 parcels per hemisphere. neuromaps
represents the parcellation in a 402-index space (201 indices per hemisphere,
including one medial-wall label). This script removes the two medial-wall
indices and converts the rotations to a zero-based 400-parcel indexing scheme.


"""

import argparse
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_annot
from neuromaps import nulls
from scipy.io import savemat


N_PARCELS_PER_HEMI = 200
N_PARCELS = 400
N_NEUROMAPS_INDICES = 402
LH_MEDIAL_WALL = 0
RH_MEDIAL_WALL = 201


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spin permutations for Schaefer-400."
    )
    parser.add_argument(
        "--lh-annot", required=True,
        help="Left-hemisphere Schaefer-400 FreeSurfer .annot file."
    )
    parser.add_argument(
        "--rh-annot", required=True,
        help="Right-hemisphere Schaefer-400 FreeSurfer .annot file."
    )
    parser.add_argument(
        "--out-dir", default=".",
        help="Output directory (default: current directory)."
    )
    parser.add_argument(
        "--n-perm", type=int, default=10000,
        help="Number of spin permutations (default: 10000)."
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Random seed passed to neuromaps (default: 1234)."
    )
    parser.add_argument(
        "--density", default="10k",
        help="fsaverage surface density used by neuromaps (default: 10k)."
    )
    return parser.parse_args()


def annot_to_labelgii(annot_path, out_path):
    """Convert a FreeSurfer .annot file to a GIFTI label file."""
    labels, _, _ = read_annot(str(annot_path))
    gifti = nib.gifti.GiftiImage()
    gifti.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            labels.astype(np.int32),
            intent="NIFTI_INTENT_LABEL",
        )
    )
    nib.save(gifti, str(out_path))
    return out_path


def validate_annotations(lh_annot, rh_annot):
    """Confirm that each hemisphere contains 200 non-medial-wall parcels."""
    lh_labels, _, _ = read_annot(str(lh_annot))
    rh_labels, _, _ = read_annot(str(rh_annot))

    lh_n = int(np.unique(lh_labels[lh_labels > 0]).size)
    rh_n = int(np.unique(rh_labels[rh_labels > 0]).size)

    print(f"LH unique nonzero labels: {lh_n} (expected {N_PARCELS_PER_HEMI})")
    print(f"RH unique nonzero labels: {rh_n} (expected {N_PARCELS_PER_HEMI})")

    if lh_n != N_PARCELS_PER_HEMI or rh_n != N_PARCELS_PER_HEMI:
        raise ValueError(
            "Unexpected number of parcels. This script expects "
            "200 Schaefer parcels per hemisphere."
        )


def convert_402_to_400(rotated):
    """
    Convert neuromaps' 402-index representation to 400 zero-based parcel indices.
    """
    rotated = np.asarray(rotated)

    if rotated.ndim != 2 or rotated.shape[1] != N_NEUROMAPS_INDICES:
        raise ValueError(
            f"Expected rotation array with shape (n_perm, 402), got {rotated.shape}."
        )

    # Valid parcel labels in the 402-index scheme:
    # LH 1..200, RH 202..401; 0 and 201 are medial wall.
    allowed = np.r_[np.arange(1, 201), np.arange(202, 402)]

    spins = np.empty((rotated.shape[0], N_PARCELS), dtype=np.int32)

    for i, idx402_full in enumerate(rotated):
        valid = (
            (idx402_full != LH_MEDIAL_WALL)
            & (idx402_full != RH_MEDIAL_WALL)
        )
        idx = np.asarray(idx402_full[valid], dtype=int)

        # If a rotated parcel lands on medial wall, fill missing labels
        # deterministically to preserve 400 entries.
        if idx.size < N_PARCELS:
            present = set(idx.tolist())
            missing = [k for k in allowed if k not in present]
            idx = np.r_[
                idx,
                np.asarray(missing[: N_PARCELS - idx.size], dtype=int),
            ]

        if idx.size > N_PARCELS:
            idx = idx[:N_PARCELS]

        if idx.size != N_PARCELS:
            raise RuntimeError(
                f"Could not construct a 400-parcel permutation for spin {i}."
            )

        lh_mask = (idx >= 1) & (idx <= 200)
        rh_mask = (idx >= 202) & (idx <= 401)

        if not np.all(lh_mask | rh_mask):
            bad = idx[~(lh_mask | rh_mask)]
            raise RuntimeError(
                f"Unexpected parcel indices after medial-wall removal: {bad}"
            )

        out = np.empty(N_PARCELS, dtype=np.int32)
        out[lh_mask] = idx[lh_mask] - 1
        out[rh_mask] = 200 + (idx[rh_mask] - 202)

        if out.min() < 0 or out.max() >= N_PARCELS:
            raise RuntimeError(
                f"Mapped indices out of range [0, 399] in permutation {i}."
            )

        spins[i] = out

    # Downstream scripts expect parcels x permutations.
    return spins.T


def main():
    args = parse_args()

    lh_annot = Path(args.lh_annot).expanduser().resolve()
    rh_annot = Path(args.rh_annot).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not lh_annot.exists():
        raise FileNotFoundError(f"Left annotation not found: {lh_annot}")
    if not rh_annot.exists():
        raise FileNotFoundError(f"Right annotation not found: {rh_annot}")

    validate_annotations(lh_annot, rh_annot)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lh_labelgii = tmpdir / "lh.schaefer400.label.gii"
        rh_labelgii = tmpdir / "rh.schaefer400.label.gii"

        annot_to_labelgii(lh_annot, lh_labelgii)
        annot_to_labelgii(rh_annot, rh_labelgii)

        print(
            f"Generating {args.n_perm} spin permutations "
            f"(atlas=fsaverage, density={args.density}, seed={args.seed})..."
        )

        base402 = np.arange(N_NEUROMAPS_INDICES, dtype=int)

        rotated = nulls.alexander_bloch(
            data=base402,
            atlas="fsaverage",
            density=args.density,
            n_perm=args.n_perm,
            seed=args.seed,
            parcellation=(str(lh_labelgii), str(rh_labelgii)),
        )

    rotated = np.asarray(rotated)

    # neuromaps may return either (402, n_perm) or (n_perm, 402).
    if rotated.shape == (N_NEUROMAPS_INDICES, args.n_perm):
        rotated = rotated.T
    elif rotated.shape == (args.n_perm, N_NEUROMAPS_INDICES):
        pass
    else:
        raise RuntimeError(
            f"Unexpected rotated shape {rotated.shape}; expected "
            f"({N_NEUROMAPS_INDICES}, {args.n_perm}) or "
            f"({args.n_perm}, {N_NEUROMAPS_INDICES})."
        )

    spins = convert_402_to_400(rotated)

    print(
        f"Final spins shape: {spins.shape}; "
        f"index range: [{spins.min()}, {spins.max()}]"
    )

    npy_path = out_dir / f"spins_400x{args.n_perm}.npy"
    mat_path = out_dir / f"spins_400x{args.n_perm}.mat"

    np.save(npy_path, spins)
    savemat(mat_path, {"spins": spins})

    print("Saved:")
    print(f"  {npy_path}")
    print(f"  {mat_path}")


if __name__ == "__main__":
    main()
