"""C17 step 1's BD-rate: two celv2enc builds on the 8 sources, q100-190, global-MSE PSNR-Y from --recon.

usage: python -I tools/c17/bd_run.py <old.exe> <new.exe> <sources dir> <work dir> [frames]
NEW_ENV=K=V[,K=V]: environment for the new build's runs only (the same exe twice for a switch).
Writes <work dir>/bd.csv (src, build, q, bytes, psnr_y, psnr_u, psnr_v) and prints each source's BD-rate (new against
old) and the mean; then the same on YUV PSNR (6 Y : 1 U : 1 V, each plane's global PSNR).
"""
import csv
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
from bdrate import bd_rate  # noqa: E402

SOURCES = [('air', 'g_air.y4m', True), ('md', 'g_md.y4m', True), ('am', 'g_am30.y4m', True), ('ck', 'g_ck.y4m', True),
           ('ab', 'g_ab.y4m', True), ('yo', 'g_yo.y4m', False), ('cr', 'g_cr.y4m', False), ('pa', 'g_pa.y4m', False)]
QS = [100, 130, 160, 190]
if os.environ.get('BD_SOURCES'):  # e.g. BD_SOURCES=s1:mxam600.y4m:1,s2air:s2_air600.y4m:1 (name:file:anime)
    SOURCES = [(n, f, a == '1') for n, f, a in (x.split(':') for x in os.environ['BD_SOURCES'].split(','))]
if os.environ.get('BD_ONLY'):  # e.g. BD_ONLY=air,ab: those sources only
    SOURCES = [s for s in SOURCES if s[0] in os.environ['BD_ONLY'].split(',')]


def y4m_frames(path, n):
    with open(path, 'rb') as f:
        head = f.readline().decode().split()
        w = int(next(t[1:] for t in head if t.startswith('W')))
        h = int(next(t[1:] for t in head if t.startswith('H')))
        size = w * h * 3 // 2
        for _ in range(n):
            if not f.readline():
                return
            yield w, h, np.frombuffer(f.read(size), np.uint8)


def global_psnr(src, recon, n):
    """Each plane's (Y, U, V) over every frame at once (one MSE), at the coded depth: an 8-bit source codes at 10 bits,
    its recon 16-bit samples (the source x4, peak 1023), as the encoder's own --psnr."""
    se, count = [0.0] * 3, [0] * 3
    frames = list(y4m_frames(src, n))
    w, h = frames[0][0], frames[0][1]
    wide = os.path.getsize(recon) == len(frames) * w * h * 3
    with open(recon, 'rb') as r:
        for w, h, yuv in frames:
            rec = np.frombuffer(r.read(w * h * 3), np.uint16) if wide else np.frombuffer(r.read(w * h * 3 // 2), np.uint8)
            o = 0
            for p, size in enumerate((w * h, w * h // 4, w * h // 4)):
                d = yuv[o:o + size].astype(np.int64) * (4 if wide else 1) - rec[o:o + size].astype(np.int64)
                se[p] += float(np.dot(d, d))
                count[p] += size
                o += size
    peak = 1023 if wide else 255
    return [99.0 if e == 0 else 10 * math.log10(peak * peak / (e / c)) for e, c in zip(se, count)]


def main():
    old, new, src_dir, work = sys.argv[1:5]
    frames = int(sys.argv[5]) if len(sys.argv) > 5 else 120
    new_env = os.environ['NEW_ENV'].split(',') if os.environ.get('NEW_ENV') else []
    if any('=' not in kv for kv in new_env):
        sys.exit(f'NEW_ENV: every element must be K=V: {os.environ["NEW_ENV"]!r}')
    os.makedirs(work, exist_ok=True)
    # The resume key (src, build, q) does not see these, so a rerun with other options must not reuse old rows.
    opts = {'old': os.path.abspath(old), 'new': os.path.abspath(new), 'frames': str(frames),
            'NEW_ARGS': os.environ.get('NEW_ARGS', ''), 'NEW_ENV': os.environ.get('NEW_ENV', '')}
    opts_path = os.path.join(work, 'opts.txt')
    if os.path.exists(opts_path):
        with open(opts_path) as f:
            prev = dict(line.rstrip('\n').split('=', 1) for line in f if '=' in line)
        diff = [f'{k}: {prev.get(k)!r} then, {v!r} now' for k, v in opts.items() if prev.get(k) != v]
        if diff:
            sys.exit(f'{work} was run with other options ({"; ".join(diff)}): use another work dir')
    else:
        with open(opts_path, 'w') as f:
            f.writelines(f'{k}={v}\n' for k, v in opts.items())
    out = os.path.join(work, 'bd.csv')
    done, fresh = set(), not os.path.exists(out)
    if not fresh:
        with open(out) as f:
            done = {(r['src'], r['build'], int(r['q'])) for r in csv.DictReader(f)}
    with open(out, 'a', newline='') as f:
        wr = csv.writer(f)
        if fresh:  # not "no rows": a rerun after a first encode failed would write the header twice
            wr.writerow(['src', 'build', 'q', 'bytes', 'psnr_y', 'psnr_u', 'psnr_v'])
        for name, file, anime in SOURCES:
            src = os.path.join(src_dir, file)
            for q in QS:
                for build, exe in (('old', old), ('new', new)):
                    if (name, build, q) in done:
                        continue
                    ivf, rec = os.path.join(work, f'{name}_{build}_{q}.ivf'), os.path.join(work, 'recon.yuv')
                    cmd = [exe, '-i', src, '-o', ivf, '--frames', str(frames), '--qindex', str(q), '--refs', '7',
                           '--workers', 'auto', '--recon', rec] + (['--tune', 'animation'] if anime else [])
                    if build == 'new' and os.environ.get('NEW_ARGS'):  # e.g. NEW_ARGS="--real-trellis 1": the new runs only
                        cmd += os.environ['NEW_ARGS'].split()
                    env = dict(os.environ)
                    if build == 'new':  # e.g. NEW_ENV=X_DBREF=1: the new runs only
                        env.update(kv.split('=', 1) for kv in new_env)
                    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL, env=env)
                    wr.writerow([name, build, q, os.path.getsize(ivf)] + [f'{v:.4f}' for v in global_psnr(src, rec, frames)])
                    f.flush()
                    os.remove(rec)
    rows = {}
    with open(out) as f:
        for r in csv.DictReader(f):
            yuv = (6 * float(r['psnr_y']) + float(r['psnr_u']) + float(r['psnr_v'])) / 8 if r.get('psnr_u') else None
            rows.setdefault(r['src'], {}).setdefault(r['build'], []).append((int(r['q']), int(r['bytes']), float(r['psnr_y']), yuv))
    for k, label in ((2, 'PSNR-Y'), (3, 'PSNR-YUV 6:1:1')):
        if k == 3 and any(t[3] is None for name, _, _ in SOURCES for b in rows[name].values() for t in b):
            break  # rows from before the chroma columns
        bds = []
        for name, _, _ in SOURCES:
            o, n_ = sorted(rows[name]['old']), sorted(rows[name]['new'])
            bd = bd_rate([t[1] for t in o], [t[k] for t in o], [t[1] for t in n_], [t[k] for t in n_])
            bds.append(bd)
            print(f'{name}: BD-rate new against old, global {label} {bd:+.3f}%')
        print(f'mean {sum(bds) / len(bds):+.3f}%, worst {max(bds):+.3f}%')


if __name__ == '__main__':
    main()
