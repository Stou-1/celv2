# C4(A): the previous anchor kept in a slot (2026-11-09)

Candidate C4(A) of docs/reports/2026-11-08-improvement-candidates.md. Against 492344e (the filter work, the same
bytes as 64d49c4), RTX 5090.

## Verdict

**Kept. BD-rate -1.46% mean, every source better (worst chika -0.55%), no CPU.**

| BD-rate, global PSNR-Y, 120 frames, q100-190, `--refs 7` | AIR | Dragon Maid | AM | chika | Angel Beats | 吉村 | crowd_run | park_joy |
|---|---|---|---|---|---|---|---|---|
| new against old | -2.14 | -0.59 | -1.99 | -0.55 | -1.76 | -0.62 | -2.09 | -1.95 |

Anime with `--tune animation`, live action with none (`tools/c17/bd_run.py`). The candidates report estimated 1.0-1.7%
from the first leaf's excess; the measured gain is about the same.

## The defect

- A reference frame took the turns' slots in coding order (`since_key_ % refs`).
- An 8-frame mini-GOP writes four references: its hidden anchor and three mids. With 7 turns, the previous anchor was
  written four writes before this one's anchor, so this group's third mid took its slot.
- The leaves are coded after the mids, so no leaf of the second group on could use the previous anchor, its nearest
  frame to the left.
- TPL (chunk_coder.cpp) and `--tplq 3`'s anchor qindex already counted on that reference.

## The change (CodecState::begin_frame)

- A reference frame now takes the turn holding the earliest frame in display order.
- Where display order is coding order (P frames, the first group after a key frame), that is the turn it took before
  (ties keep it), so those frames' bytes do not move.
- In a mini-GOP the previous anchor is the latest of the old group in display order, so it is evicted last.
- This holds for 16-frame groups too (an anchor and 5 mids: 6 writes of 7 turns).
- `FrameSpec::refresh` (X_L0) still overrides it.

## Reproduce

```bash
python -I tools/c17/bd_run.py old.exe new.exe sources work 120
```

## G2.1 on top: skip-mode trials predicted with the SHARP filter

The decoder predicts skip mode with MULTITAP_SHARP (set_default_interp_filters()); BlockWriter::choose_skip_mode()
scored the candidates with the frame's filter. One line. Against C4(A), the same runs: **-0.09% mean** (AM -0.38,
Angel Beats -0.21, park_joy -0.16, crowd_run -0.13, AIR +0.23%); S1 CPU a frame 1979 -> 1982 ms (2 rounds each).
Kept: the scoring now matches what is coded.
