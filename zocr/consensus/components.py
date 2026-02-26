"""Connected-component helpers for consensus runtime consumers."""
from __future__ import annotations

from collections.abc import Sequence

__all__ = ["_rle_runs", "_cc_label_rle"]


def _rle_runs(binary: Sequence[Sequence[int]]):
    """Yield run-length encoded spans for each row of ``binary``."""

    H = len(binary)
    runs_by_row = []
    for y in range(H):
        row = binary[y]
        width = len(row)
        runs = []
        in_run = False
        start = 0
        for x in range(width):
            v = row[x]
            if v and not in_run:
                in_run = True
                start = x
            elif (not v) and in_run:
                runs.append((start, x))
                in_run = False
        if in_run:
            runs.append((start, width))
        runs_by_row.append(runs)
    return runs_by_row


def _cc_label_rle(binary: Sequence[Sequence[int]]):
    """Return bounding boxes for connected components via RLE linking."""

    runs_by_row = _rle_runs(binary)
    parent = []
    bbox = []
    lab_of_run = []
    row_offsets = [0]
    for y, runs in enumerate(runs_by_row):
        row_offsets.append(row_offsets[-1] + len(runs))
        for (x0, x1) in runs:
            lab = len(parent)
            parent.append(lab)
            bbox.append([x0, y, x1, y + 1, x1 - x0])
            lab_of_run.append(lab)
        if y == 0:
            continue
        prev_runs = runs_by_row[y - 1]
        if not runs or not prev_runs:
            continue
        i = 0
        j = 0
        while i < len(prev_runs) and j < len(runs):
            p0, p1 = prev_runs[i]
            c0, c1 = runs[j]
            if p1 <= c0:
                i += 1
            elif c1 <= p0:
                j += 1
            else:
                rp = _find(lab_of_run[row_offsets[y - 1] + i], parent)
                rc = _find(lab_of_run[row_offsets[y] + j], parent)
                if rp != rc:
                    parent[rc] = rp
                    bpr, bcr = bbox[rp], bbox[rc]
                    bpr[0] = min(bpr[0], bcr[0])
                    bpr[1] = min(bpr[1], bcr[1])
                    bpr[2] = max(bpr[2], bcr[2])
                    bpr[3] = max(bpr[3], bcr[3])
                    bpr[4] += bcr[4]
                if p1 < c1:
                    i += 1
                else:
                    j += 1
    out = {}
    idx = 0
    for y, runs in enumerate(runs_by_row):
        for (x0, x1) in runs:
            r = _find(lab_of_run[idx], parent)
            idx += 1
            if r not in out:
                out[r] = [x0, y, x1, y + 1, (x1 - x0)]
            else:
                b = out[r]
                if x0 < b[0]:
                    b[0] = x0
                if y < b[1]:
                    b[1] = y
                if x1 > b[2]:
                    b[2] = x1
                if y + 1 > b[3]:
                    b[3] = y + 1
                b[4] += (x1 - x0)
    return [tuple(v) for v in out.values()]


def _find(idx: int, parent: list) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx
