# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if eps < 0:
        raise ValueError("eps должен быть неотрицательным")

    if tt.order == 1:
        return tt.copy()

    rounded = right_canonicalize(tt, backend)
    cores = []
    for core in rounded.cores:
        cores.append(backend.copy(core))

    first_norm = backend.norm(cores[0])
    if first_norm > 1e-30:
        delta = eps * first_norm / math.sqrt(tt.order - 1)
    else:
        delta = 0.0

    for k in range(tt.order - 1):
        core = cores[k]
        left_rank, mode_size, right_rank = core.shape
        matrix = backend.reshape(core, (left_rank * mode_size, right_rank))
        U, S, Vt = backend.svd(matrix, full_matrices=False)
        rank = _compute_rank(S, delta, max_rank)

        U_part = _truncate_columns(U, rank, backend)
        S_part = _truncate_vector(S, rank, backend)
        Vt_part = _truncate_rows(Vt, rank, backend)

        cores[k] = backend.reshape(U_part, (left_rank, mode_size, rank))
        transfer = _multiply_diag_matrix(S_part, Vt_part, rank, backend)

        next_core = cores[k + 1]
        _, next_mode_size, next_right_rank = next_core.shape
        new_next = backend.zeros((rank, next_mode_size, next_right_rank))

        for a in range(rank):
            for i in range(next_mode_size):
                for b in range(next_right_rank):
                    value = 0.0
                    for c in range(right_rank):
                        value += transfer[a, c] * next_core[c, i, b]
                    new_next[a, i, b] = value

        cores[k + 1] = new_next

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть вектором")

    rank = S.shape[0]
    if rank < 1:
        return 1

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
