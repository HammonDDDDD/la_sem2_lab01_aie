# core/utils.py

"""Вспомогательные функции для работы с тензорами."""


def validate_shape(
    shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.

    Убеждается, что shape является последовательностью положительных целых
    чисел. Преобразует список в кортеж для единообразия.

    Args:
        shape: кортеж или список размеров тензора по каждой моде

    Returns:
        tuple: проверенный кортеж положительных целых чисел

    Raises:
        TypeError:  если shape не является tuple или list
        ValueError: если хотя бы один элемент shape не является
                    положительным целым числом
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape должен быть tuple или list")

    if len(shape) == 0:
        raise ValueError("shape не должен быть пустым")

    result = []
    for value in shape:
        if type(value) is not int:
            raise ValueError("размеры shape должны быть целыми числами")
        if value <= 0:
            raise ValueError("размеры shape должны быть положительными")
        result.append(value)

    return tuple(result)


def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    size = 1
    for value in shape:
        size *= value
    return size


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает кортеж strides, содержащий для каждой моды k свой strides[k].

    Stride по моде k — это число элементов в плоском списке, на которое
    нужно сдвинуться, чтобы перейти к следующему элементу вдоль моды k.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    strides = [1] * len(shape)
    current = 1

    for i in range(len(shape) - 1, -1, -1):
        strides[i] = current
        current *= shape[i]

    return tuple(strides)


def multi_index_to_flat(
    multi_index: tuple[int, ...],
    strides: tuple[int, ...]
) -> int:
    """
    Возвращает позицию элемента в плоском списке данных по его
    многомерным координатам и заранее вычисленным strides.

    Args:
        multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1})
        strides:     кортеж шагов   (s_0, s_1, ..., s_{d-1})
    """
    if len(multi_index) != len(strides):
        raise ValueError("длина индекса не совпадает с размерностью")

    flat_index = 0
    for i in range(len(multi_index)):
        if type(multi_index[i]) is not int:
            raise ValueError("индексы должны быть целыми числами")
        if multi_index[i] < 0:
            raise IndexError("индекс вне границ тензора")
        flat_index += multi_index[i] * strides[i]

    return flat_index


def flat_to_multi_index(
    flat_index: int,
    shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Возвращает мультииндекс на основе плоского индекса.

    Args:
        flat_index: плоский индекс в списке данных
        shape:      кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    if type(flat_index) is not int:
        raise ValueError("плоский индекс должен быть целым числом")

    size = compute_size(shape)
    if flat_index < 0 or flat_index >= size:
        raise IndexError("индекс вне границ тензора")

    strides = compute_strides(shape)
    values = []
    rest = flat_index

    for stride in strides:
        values.append(rest // stride)
        rest = rest % stride

    return tuple(values)

def check_shapes_match(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...]
) -> None:
    """
    Проверяет совпадение форм двух тензоров.

    Используется перед поэлементными операциями (сложение, вычитание),
    чтобы гарантировать совместимость тензоров.

    Args:
        shape1: кортеж размеров первого тензора
        shape2: кортеж размеров второго тензора

    Raises:
        ValueError: если формы не совпадают
    """
    if shape1 != shape2:
        raise ValueError("формы тензоров не совпадают")
