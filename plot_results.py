#!/usr/bin/env python3
"""Plot the checked-in historical aggregate measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def main():
    data = json.loads((ROOT / 'results/a100-concurrency.json').read_text())
    colors = ['#e95420', '#8b72c8', '#3d9181']
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for series, color in zip(data['series'], colors):
        rows = series['levels']
        ax.plot([r['concurrency'] for r in rows], [r['aggregate_tokens_per_s'] for r in rows],
                marker='o', color=color, linewidth=2, label=series['label'])
    ax.set(xlabel='Concurrent requests per historical batch', ylabel='Aggregate output tokens / second',
           title='A100 40 GB — historical serving configurations (July 2026)')
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='upper left')
    fig.text(.5, -.035, 'Different models / quantization: descriptive measurements, not a controlled framework ranking.',
             ha='center', fontsize=9)
    fig.savefig(ROOT / 'results/a100-throughput.svg', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
