"""Plot the installed-capacity (kWp) distribution of uk_pv systems.

Top panel: every system in metadata.csv within the library's 0.5-30 kWp filter.
Bottom panel: systems that survived build_pv_library.py's quality filters (kwp in the npz).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SNAP = glob.glob('/workspace/.hf_home/hub/datasets--openclimatefix--uk_pv/snapshots/*/')[0]
INK, MUTED, GRID, BAR = '#1f2328', '#656d76', '#e6e8eb', '#2f6fde'


def panel(ax, kwp, title, bins, xmax):
    ax.hist(np.minimum(kwp, xmax), bins=bins, color=BAR, edgecolor='white', linewidth=0.8)
    q1, med, q3 = np.percentile(kwp, [25, 50, 75])
    ax.axvline(med, color=INK, lw=1, ls='--')
    ax.text(0.02, 0.95, f'median {med:.2f} kWp (dashed)\nIQR {q1:.2f}–{q3:.2f} kWp', transform=ax.transAxes,
            va='top', ha='left', fontsize=9, color=INK)
    over = (kwp > xmax).sum()
    note = f'n = {len(kwp):,}' + (f'  ·  {over:,} above {xmax:g} kWp in last bin' if over else '')
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.text(1, 1.02, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=MUTED)
    ax.set_ylabel('systems', color=MUTED)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/workspace/data/pv_library_15min.npz')
    ap.add_argument('--out', default='/workspace/plots/kwp_distribution.png')
    ap.add_argument('--xmax', type=float, default=6.0)
    ap.add_argument('--bin', type=float, default=0.1)
    args = ap.parse_args()

    meta = pl.read_csv(os.path.join(SNAP, 'metadata.csv'))['kWp'].drop_nulls().to_numpy()
    meta = meta[(meta >= 0.5) & (meta <= 30)]
    lib = np.load(args.npz)['kwp']
    bins = np.arange(0, args.xmax + args.bin, args.bin)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    panel(axes[0], meta, 'All uk_pv systems (0.5–30 kWp)', bins, args.xmax)
    panel(axes[1], lib, 'PV library (after quality filters)', bins, args.xmax)
    axes[1].set_xlabel('installed capacity (kWp)', color=MUTED)
    axes[1].set_xlim(0, args.xmax)
    fig.suptitle('Installed PV capacity distribution', x=0.01, ha='left', fontsize=13, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    for name, a in (('metadata', meta), ('library', lib)):
        print(name, len(a), np.percentile(a, [1, 5, 25, 50, 75, 95, 99]).round(2))
    print('saved', args.out)


if __name__ == '__main__':
    main()
