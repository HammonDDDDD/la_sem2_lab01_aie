# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

import random

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not isinstance(cores, list):
            raise TypeError("cores должен быть списком")
        if len(cores) == 0:
            raise ValueError("список ядер не должен быть пустым")

        for core in cores:
            if not isinstance(core, DenseTensor):
                raise TypeError("каждое ядро должно быть DenseTensor")
            if core.ndim != 3:
                raise ValueError("каждое ядро должно быть трехмерным")

        if cores[0].shape[0] != 1:
            raise ValueError("первый TT-ранг должен быть равен 1")
        if cores[-1].shape[2] != 1:
            raise ValueError("последний TT-ранг должен быть равен 1")

        for i in range(len(cores) - 1):
            if cores[i].shape[2] != cores[i + 1].shape[0]:
                raise ValueError("соседние TT-ранги не совпадают")

        self.cores = []
        for core in cores:
            self.cores.append(core.copy())

        self.order = len(self.cores)

        shape = []
        ranks = [self.cores[0].shape[0]]
        for core in self.cores:
            shape.append(core.shape[1])
            ranks.append(core.shape[2])

        self.shape = tuple(shape)
        self.ranks = tuple(ranks)


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        shape = validate_shape(shape)
        order = len(shape)

        if not isinstance(ranks, (tuple, list)):
            raise TypeError("ranks должен быть tuple или list")

        ranks = list(ranks)
        if len(ranks) == order - 1:
            ranks = [1] + ranks + [1]
        elif len(ranks) != order + 1:
            raise ValueError("неверное число TT-рангов")

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("граничные TT-ранги должны быть равны 1")

        for rank in ranks:
            if type(rank) is not int or rank <= 0:
                raise ValueError("TT-ранги должны быть положительными целыми")

        rng = random.Random(seed)
        cores = []
        for i in range(order):
            core_shape = (ranks[i], shape[i], ranks[i + 1])
            size = compute_size(core_shape)
            data = []
            for _ in range(size):
                data.append(rng.uniform(-1.0, 1.0))
            cores.append(DenseTensor(core_shape, data=data))

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if isinstance(indices, list):
            indices = tuple(indices)
        if not isinstance(indices, tuple):
            raise TypeError("indices должен быть tuple или list")
        if len(indices) != self.order:
            raise ValueError("длина индекса не совпадает с порядком тензора")

        for i in range(self.order):
            if type(indices[i]) is not int:
                raise ValueError("индексы должны быть целыми числами")
            if indices[i] < 0 or indices[i] >= self.shape[i]:
                raise IndexError("индекс вне границ тензора")

        values = [1.0]
        for k in range(self.order):
            core = self.cores[k]
            left_rank, _, right_rank = core.shape
            new_values = [0.0] * right_rank
            for right in range(right_rank):
                total = 0.0
                for left in range(left_rank):
                    total += values[left] * core[left, indices[k], right]
                new_values[right] = total
            values = new_values

        return values[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = DenseTensor.zeros(self.shape)
        for flat_index in range(result.size):
            index = flat_to_multi_index(flat_index, self.shape)
            result.data[flat_index] = self.get_element(index)
        return result

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        result = []
        for core in self.cores:
            result.append(core.shape)
        return result

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        total = 0
        for core in self.cores:
            total += core.size
        return total

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        return compute_size(self.shape) / self.total_storage()

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        cores = []
        for core in self.cores:
            cores.append(core.copy())
        return TTTensor(cores)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        return (
            "TTTensor("
            f"order={self.order}, "
            f"shape={self.shape}, "
            f"ranks={self.ranks}, "
            f"cores={self.core_sizes()}, "
            f"storage={self.total_storage()})"
        )

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()
