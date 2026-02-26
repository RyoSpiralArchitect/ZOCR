import pytest

pytest.importorskip("numpy")


def test_bm25_numba_matches_python() -> None:
    import numpy as np

    from zocr.core.indexer import _bm25_numba_score, _bm25_py_score

    rng = np.random.default_rng(0)
    V = 64
    N = 200
    df = rng.integers(1, N, size=V, dtype=np.int32)

    for _ in range(25):
        dl = int(rng.integers(1, 40))
        doc_ids = rng.integers(0, V, size=dl, dtype=np.int32)
        doc_arr = np.concatenate([doc_ids, np.array([-1], dtype=np.int32)])

        q_len = int(rng.integers(0, 12))
        if q_len == 0:
            q_arr = np.array([-1], dtype=np.int32)
        else:
            q_arr = rng.integers(0, V, size=q_len, dtype=np.int32)

        avgdl = float(rng.integers(1, 60))

        numba_score = float(_bm25_numba_score(N, avgdl, df, dl, q_arr, doc_arr))
        py_score = float(_bm25_py_score(N, avgdl, df, dl, q_arr, doc_arr))

        assert abs(numba_score - py_score) < 1e-6


def test_bm25_scores_all_matches_single_doc() -> None:
    import numpy as np

    from zocr.core.indexer import (
        _bm25_numba_score,
        _bm25_numba_scores_all,
        _bm25_numba_scores_all_parallel,
    )

    rng = np.random.default_rng(0)
    V = 64
    N = 250
    df = rng.integers(1, N, size=V, dtype=np.int32)

    lengths = rng.integers(0, 60, size=N, dtype=np.int32)
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    flat = rng.integers(0, V, size=int(offsets[-1]), dtype=np.int32)

    for _ in range(15):
        q_len = int(rng.integers(0, 12))
        if q_len == 0:
            q_arr = np.array([-1], dtype=np.int32)
        else:
            q_arr = rng.integers(0, V, size=q_len, dtype=np.int32)
        avgdl = float(rng.integers(1, 80))

        all_scores = _bm25_numba_scores_all(N, avgdl, df, lengths, offsets, flat, q_arr)
        if _bm25_numba_scores_all_parallel is not None:
            par_scores = _bm25_numba_scores_all_parallel(
                N, avgdl, df, lengths, offsets, flat, q_arr
            )
            assert np.max(np.abs(all_scores - par_scores)) < 1e-6

        for i in rng.choice(N, size=25, replace=False):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            doc_ids = flat[start:end]
            dl = int(lengths[i])
            doc_arr = np.concatenate([doc_ids, np.array([-1], dtype=np.int32)])
            single = float(_bm25_numba_score(N, avgdl, df, dl, q_arr, doc_arr))
            assert abs(float(all_scores[i]) - single) < 1e-6
