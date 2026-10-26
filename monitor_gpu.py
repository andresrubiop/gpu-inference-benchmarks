#!/usr/bin/env python3
"""Sample one selected NVIDIA GPU to CSV until interrupted."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--output', default='results/gpu.csv')
    args = parser.parse_args()
    if args.gpu < 0 or args.interval <= 0 or args.samples < 1:
        parser.error('gpu must be nonnegative; interval and samples must be positive')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp_utc', 'gpu_index', 'power_w', 'temperature_c', 'utilization_pct', 'memory_used_mib'])
        for _ in range(args.samples):
            result = subprocess.run(['nvidia-smi', '--id=' + str(args.gpu),
                                     '--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
            row = next(csv.reader(result.stdout.splitlines()))
            if len(row) != 4:
                raise RuntimeError('Unexpected nvidia-smi columns')
            writer.writerow([datetime.now(timezone.utc).isoformat(), args.gpu, *[value.strip() for value in row]])
            handle.flush()
            time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
