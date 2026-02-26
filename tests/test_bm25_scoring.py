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

