"""Execution-core helpers shared across the consensus runtime stack."""
from .utils import clamp
from .binarization import (
    _box_mean,
    _binarize_pure,
    _estimate_slant_slope,
    _shear_rows_binary,
    _suppress_diagonal_bridges,
    _apply_italic_guard,
)
from .morphology import _dilate_binary_rect
from .components import _rle_runs, _cc_label_rle
from .columns import (
    _dp_means_1d,
    _btree_partition,
    _btree_column_centers,
    _vertical_vote_boundaries,
    _smooth_per_column,
)

__all__ = [
    "clamp",
    "_box_mean",
    "_binarize_pure",
    "_estimate_slant_slope",
    "_shear_rows_binary",
    "_suppress_diagonal_bridges",
    "_apply_italic_guard",
    "_dilate_binary_rect",
    "_rle_runs",
    "_cc_label_rle",
    "_dp_means_1d",
    "_btree_partition",
    "_btree_column_centers",
    "_vertical_vote_boundaries",
    "_smooth_per_column",
]
