# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if not isinstance(tensor, DenseTensor):
        raise TypeError("tensor должен быть DenseTensor")
    if eps < 0:
        raise ValueError("eps должен быть неотрицательным")

    d = tensor.ndim
    shape = tensor.shape

    if d == 1:
        core = backend.reshape(tensor, (1, shape[0], 1))
        return TTTensor([core])

    norm = backend.norm(tensor)
    if norm > 1e-30:
        delta = eps * norm / math.sqrt(d - 1)
    else:
        delta = 0.0

    cores = []
    current = backend.copy(tensor)
    prev_rank = 1

    for k in range(d - 1):
        rows = prev_rank * shape[k]
        cols = 1
        for i in range(k + 1, d):
            cols *= shape[i]

        matrix = backend.reshape(current, (rows, cols))
        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _compute_truncated_rank(S, delta, max_rank)

        U_part = _truncate_columns(U, rank, backend)
        core = backend.reshape(U_part, (prev_rank, shape[k], rank))
        cores.append(core)

        S_part = _truncate_vector(S, rank, backend)
        Vt_part = _truncate_rows(Vt, rank, backend)
        current = _multiply_diag_matrix(S_part, Vt_part, rank, backend)
        prev_rank = rank

    last_core = backend.reshape(current, (prev_rank, shape[-1], 1))
    cores.append(last_core)

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть вектором")

    count = S.shape[0]
    if count == 0:
        return 1

    limit = max(1e-12, 1e-8 * abs(S.data[0]))
    rank = 0
    for i in range(count):
        if abs(S.data[i]) > limit:
            rank = i + 1

    if rank < 1:
        rank = 1

    if delta > 0:
        tail = 0.0
        while rank > 1 and tail + S.data[rank - 1] * S.data[rank - 1] <= delta * delta:
            tail += S.data[rank - 1] * S.data[rank - 1]
            rank -= 1

    if max_rank is not None:
        rank = min(rank, max_rank)

    if rank < 1:
        rank = 1

    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным")
    rows, cols = matrix.shape
    if rank < 1 or rank > cols:
        raise ValueError("неверный rank")

    result = backend.zeros((rows, rank))
    for i in range(rows):
        for j in range(rank):
            result[i, j] = matrix[i, j]
    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным")
    rows, cols = matrix.shape
    if rank < 1 or rank > rows:
        raise ValueError("неверный rank")

    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[i, j] = matrix[i, j]
    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError("vector должен быть одномерным")
    if rank < 1 or rank > vector.shape[0]:
        raise ValueError("неверный rank")

    result = backend.zeros((rank,))
    for i in range(rank):
        result[i] = vector[i]
    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1 or matrix.ndim != 2:
        raise ValueError("неверные размерности")
    if diag_vec.shape[0] != rank or matrix.shape[0] != rank:
        raise ValueError("rank не совпадает с размерами")

    cols = matrix.shape[1]
    result = backend.zeros((rank, cols))
    for i in range(rank):
        for j in range(cols):
            result[i, j] = diag_vec[i] * matrix[i, j]
    return result
