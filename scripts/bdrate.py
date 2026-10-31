"""BD-rate of celv2 against AVM from bench_vs_avm.ps1's CSV.

Usage: python -I scripts/bdrate.py results.csv      (prints BD-rate and speed ratio per resolution)
       python -I scripts/bdrate.py results.csv --anchor base   (every encoder vs encoder 'base')
       python -I scripts/bdrate.py --selftest
Standard Bjontegaard: cubic polynomial of PSNR -> log(rate), integrated over the overlapping PSNR
range. Negative = celv2 needs fewer bits than AVM for the same PSNR.
"""
import csv
import math
import sys
from collections import defaultdict

import numpy as np


def bd_rate(anchor_rate, anchor_psnr, test_rate, test_psnr):
    pa = np.polyfit(anchor_psnr, np.log(anchor_rate), 3)
    pt = np.polyfit(test_psnr, np.log(test_rate), 3)
    lo = max(min(anchor_psnr), min(test_psnr))
    hi = min(max(anchor_psnr), max(test_psnr))
    if hi <= lo:
        return float('nan')
    ia, it = np.polyint(pa), np.polyint(pt)
    avg = ((np.polyval(it, hi) - np.polyval(it, lo)) - (np.polyval(ia, hi) - np.polyval(ia, lo))) / (hi - lo)
    return (math.exp(avg) - 1) * 100


def selftest():
    r = [100, 200, 400, 800]
    p = [30, 33, 36, 39]
    assert abs(bd_rate(r, p, r, p)) < 1e-9
    assert abs(bd_rate(r, p, [2 * x for x in r], p) - 100) < 1e-6  # double rate -> +100%
    print('selftest ok')


def main(path, anchor=None):
    rows = defaultdict(lambda: defaultdict(list))
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows[r['clip']][r['encoder']].append(r)
    rate = lambda rs: [float(r['kbps']) for r in rs]
    psnr = lambda rs: [float(r['psnr_y']) for r in rs]
    total = lambda rs, k: sum(float(r[k]) for r in rs)
    for clip, enc in rows.items():
        if anchor:  # every encoder against the anchor, worst BD-rate first
            a = enc[anchor]
            res = [(bd_rate(rate(a), psnr(a), rate(t), psnr(t)), name, t)
                   for name, t in enc.items() if name != anchor]
            for bd, name, t in sorted(res, key=lambda x: -x[0]):
                print(f"{clip},{name},{bd:+.1f}%,wall {total(t, 'seconds') / total(a, 'seconds'):.3f},"
                      f"cpu {total(t, 'cpu_seconds') / total(a, 'cpu_seconds'):.3f}")
            continue
        a, g = enc['avm'], enc['celv2']
        bd = bd_rate(rate(a), psnr(a), rate(g), psnr(g))
        print(f"{clip}: BD-rate(celv2 vs AVM, PSNR-Y) = {bd:+.1f}%, "
              f"wall celv2/AVM = {total(g, 'seconds') / total(a, 'seconds'):.4f} "
              f"({total(g, 'seconds'):.1f} s vs {total(a, 'seconds'):.1f} s), "
              f"CPU celv2/AVM = {total(g, 'cpu_seconds') / total(a, 'cpu_seconds'):.4f}")


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    elif len(sys.argv) == 4 and sys.argv[2] == '--anchor':
        main(sys.argv[1], sys.argv[3])
    else:
        main(sys.argv[1])
