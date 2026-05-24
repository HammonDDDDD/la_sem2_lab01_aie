# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [fill] * self.size
        else:
            if len(data) != self.size:
                raise ValueError("размер data не совпадает с shape")
            self.data = list(data)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        rng = random.Random(seed)
        checked_shape = validate_shape(shape)
        size = compute_size(checked_shape)
        data = []

        for _ in range(size):
            if integer:
                data.append(rng.randint(low, high))
            else:
                data.append(rng.uniform(low, high))

        return DenseTensor(checked_shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        if not isinstance(nested, (list, tuple)):
            raise TypeError("nested должен быть списком")

        def find_shape(value):
            if not isinstance(value, (list, tuple)):
                return ()
            if len(value) == 0:
                raise ValueError("пустые списки не задают тензор")

            first_shape = find_shape(value[0])
            for item in value:
                if find_shape(item) != first_shape:
                    raise ValueError("вложенный список должен быть прямоугольным")
            return (len(value),) + first_shape

        def add_values(value, result):
            if isinstance(value, (list, tuple)):
                for item in value:
                    add_values(item, result)
            else:
                result.append(value)

        shape = find_shape(nested)
        data = []
        add_values(nested, data)
        return DenseTensor(shape, data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if type(multi_index) is int:
            return flat_to_multi_index(multi_index, self.shape)

        if isinstance(multi_index, list):
            multi_index = tuple(multi_index)

        if not isinstance(multi_index, tuple):
            raise TypeError("индекс должен быть int, tuple или list")

        if len(multi_index) != self.ndim:
            raise IndexError("неверная длина индекса")

        for i in range(self.ndim):
            value = multi_index[i]
            if type(value) is not int:
                raise IndexError("индексы должны быть целыми числами")
            if value < 0 or value >= self.shape[i]:
                raise IndexError("индекс вне границ тензора")

        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(index, self.strides)
        return self.data[flat_index]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(index, self.strides)
        self.data[flat_index] = value

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        checked_shape = validate_shape(new_shape)
        if compute_size(checked_shape) != self.size:
            raise ValueError("новая форма должна сохранять число элементов")
        return DenseTensor(checked_shape, data=self.data[:])

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if type(mode) is not int:
            raise TypeError("mode должен быть целым числом")
        if mode < 0 or mode >= self.ndim:
            raise ValueError("mode вне границ тензора")

        row_count = self.shape[mode]
        col_count = self.size // row_count
        result = DenseTensor.zeros((row_count, col_count))

        other_shape = []
        for i in range(self.ndim):
            if i != mode:
                other_shape.append(self.shape[i])
        other_strides = compute_strides(tuple(other_shape)) if len(other_shape) > 0 else ()

        for flat_index in range(self.size):
            index = flat_to_multi_index(flat_index, self.shape)
            row = index[mode]
            other_index = []
            for i in range(self.ndim):
                if i != mode:
                    other_index.append(index[i])
            if len(other_index) == 0:
                col = 0
            else:
                col = multi_index_to_flat(tuple(other_index), other_strides)
            result[row, col] = self.data[flat_index]

        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if type(k) is not int:
            raise TypeError("k должен быть целым числом")
        if k < 0 or k >= self.ndim - 1:
            raise ValueError("k вне границ тензора")

        left = 1
        for i in range(k + 1):
            left *= self.shape[i]

        right = self.size // left
        return self.reshape((left, right))

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data[:])

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        value = 0.0
        for item in self.data:
            value += item * item
        return math.sqrt(value)

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            return NotImplemented
        check_shapes_match(self.shape, other.shape)
        data = []
        for i in range(self.size):
            data.append(self.data[i] + other.data[i])
        return DenseTensor(self.shape, data=data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            return NotImplemented
        check_shapes_match(self.shape, other.shape)
        data = []
        for i in range(self.size):
            data.append(self.data[i] - other.data[i])
        return DenseTensor(self.shape, data=data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        data = []
        for item in self.data:
            data.append(item * scalar)
        return DenseTensor(self.shape, data=data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self * -1

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if not hasattr(other, "shape") or not hasattr(other, "data"):
            return False
        if self.shape != other.shape:
            return False

        for i in range(self.size):
            a = self.data[i]
            b = other.data[i]
            limit = atol + rtol * max(abs(a), abs(b))
            if abs(a - b) > limit:
                return False

        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        def build(level, offset):
            if level == self.ndim - 1:
                values = []
                for i in range(self.shape[level]):
                    values.append(self.data[offset + i])
                return values

            values = []
            step = self.strides[level]
            for i in range(self.shape[level]):
                values.append(build(level + 1, offset + i * step))
            return values

        return build(0, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.data})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()
