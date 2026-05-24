# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("формы TT-тензоров не совпадают")

    if tt1.order == 1:
        core = backend.add(tt1.cores[0], tt2.cores[0])
        return TTTensor([core])

    cores = []

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        left1, mode_size, right1 = core1.shape
        left2, _, right2 = core2.shape

        if k == 0:
            new_core = backend.zeros((1, mode_size, right1 + right2))
            for i in range(mode_size):
                for b in range(right1):
                    new_core[0, i, b] = core1[0, i, b]
                for b in range(right2):
                    new_core[0, i, right1 + b] = core2[0, i, b]
        elif k == tt1.order - 1:
            new_core = backend.zeros((left1 + left2, mode_size, 1))
            for a in range(left1):
                for i in range(mode_size):
                    new_core[a, i, 0] = core1[a, i, 0]
            for a in range(left2):
                for i in range(mode_size):
                    new_core[left1 + a, i, 0] = core2[a, i, 0]
        else:
            new_core = backend.zeros((left1 + left2, mode_size, right1 + right2))
            for a in range(left1):
                for i in range(mode_size):
                    for b in range(right1):
                        new_core[a, i, b] = core1[a, i, b]
            for a in range(left2):
                for i in range(mode_size):
                    for b in range(right2):
                        new_core[left1 + a, i, right1 + b] = core2[a, i, b]

        cores.append(new_core)

    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    cores = []
    for core in tt.cores:
        cores.append(backend.copy(core))
    cores[0] = backend.scale(cores[0], alpha)
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("формы TT-тензоров не совпадают")

    cores = []
    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        left1, mode_size, right1 = core1.shape
        left2, _, right2 = core2.shape
        new_core = backend.zeros((left1 * left2, mode_size, right1 * right2))

        for a1 in range(left1):
            for a2 in range(left2):
                new_left = a1 * left2 + a2
                for i in range(mode_size):
                    for b1 in range(right1):
                        for b2 in range(right2):
                            new_right = b1 * right2 + b2
                            new_core[new_left, i, new_right] = (
                                core1[a1, i, b1] * core2[a2, i, b2]
                            )

        cores.append(new_core)

    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("формы TT-тензоров не совпадают")

    current = backend.ones((1, 1))

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]
        left1, mode_size, right1 = core1.shape
        left2, _, right2 = core2.shape
        new_current = backend.zeros((right1, right2))

        for b1 in range(right1):
            for b2 in range(right2):
                value = 0.0
                for a1 in range(left1):
                    for a2 in range(left2):
                        for i in range(mode_size):
                            value += (
                                current[a1, a2]
                                * core1[a1, i, b1]
                                * core2[a2, i, b2]
                            )
                new_current[b1, b2] = value

        current = new_current

    return current[0, 0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    value = tt_dot(tt, tt, backend)
    if value < 0:
        value = 0.0
    return math.sqrt(value)


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("формы TT-тензоров не совпадают")

    norm1 = tt_dot(tt1, tt1, backend)
    norm2 = tt_dot(tt2, tt2, backend)
    mixed = tt_dot(tt1, tt2, backend)
    value = norm1 + norm2 - 2.0 * mixed
    scale = max(abs(norm1), abs(norm2), abs(mixed), 1.0)

    if abs(value) <= 1e-14 * scale:
        value = 0.0
    if value < 0:
        value = 0.0

    return math.sqrt(value)
