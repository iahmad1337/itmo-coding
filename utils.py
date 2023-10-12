import numpy as np
import numpy.typing as npt
from typing import Iterable, Iterator, Any, TypeVar
import galois
import itertools
import typeguard
import heapq

MATRIX_ELEMS: {str} = {'0', '1', '\n', ' '}

def string_to_matrix(s: str) -> [[int]]:
    assert all(c in MATRIX_ELEMS for c in s), "String must only contain 1s, 0s and whitespace"
    return [[int(c) for c in line] for line in s.split()]


def matrix_to_string(m: Iterable[Iterable[int]]) -> str:
    return '\n'.join(''.join(str(c) for c in line) for line in m)


def compare_rows(a: Iterable[int], b: Iterable[int]) -> int:
    if np.count_nonzero(a) == 0:
        return 0 if np.count_nonzero(b) == 0 else 1
    if np.count_nonzero(b) == 0:
        return -1
    return np.argmax(a) - np.argmax(b)

def is_row_echelon_form(matrix: Iterable[Iterable[int]]) -> bool:
    if len(matrix) < 2:
        return True
    return np.all([compare_rows(lhs, rhs) <= 0 for lhs, rhs in zip(matrix[:-1], matrix[1:])])

@typeguard.typechecked
def row_echelon_form(matrix: npt.ArrayLike) -> np.ndarray:
    '''
    Make a "stairy" matrix via gaussian elimination. Actually, this function builds *reduced* row echelon form (a.k.a. row canonical form) - and it is unique!
    :param matrix: matrix
    :return: matrix
    '''
    # Make a "stairy" matrix via gaussian elimination
    result = galois.GF2(matrix, copy=True)
    used = np.zeros(result.shape[0])
    for col in range(result.shape[1]):
        for row in range(result.shape[0]):
            if not used[row] and result[row, col] == 1:
                break
        pivot_row = row
        used[pivot_row] = True
        if pivot_row == result.shape[0] or result[pivot_row, col] == 0:
            continue
        for row in range(result.shape[0]):
            if row != pivot_row and result[row, col] == 1:
                result[row] += result[pivot_row]
    # sort rows based on the b(row)
    for _ in range(result.shape[0]):
        for j in range(result.shape[0] - 1):
            if compare_rows(result[j], result[j + 1]) > 0:
                result[[j, j + 1]] = result[[j + 1, j]]
                pass
    return result


def subsets(items: Iterable[Any]) -> Iterator[Iterable[Any]]:
    yield tuple()
    for i in range(len(items)):
        for j in itertools.combinations(items, i + 1):
            yield j

def get_generator_matrix(vectors: galois.GF2) -> galois.GF2:
    '''
    Get generator matrix for linear block code
    :param vectors: vectors that are the elements of the linear block code
    :return: generator matrix
    '''
    rfe = row_echelon_form(vectors)
    # return all rows except those that are fully zero wrapped in numpy array
    np_rfe = np.array(rfe)
    return rfe[np.count_nonzero(np_rfe, axis=1) != 0, :]

def get_code_by_generator_matrix(generator_matrix: galois.GF2) -> galois.GF2:
    '''
    Get all codes by generator matrix
    :param generator_matrix: generator matrix
    :return: all code of this linear block code
    '''
    result = []
    for s in subsets(generator_matrix):
        c = galois.GF2(np.zeros(generator_matrix.shape[1], dtype=np.int8))
        for row in s:
            c += row
        result.append(c)
    return galois.GF2(result)

def get_code_minimal_distance(vectors: galois.GF2) -> int:
    hemming_weights = [np.count_nonzero(row) for row in vectors]
    return heapq.nsmallest(2, hemming_weights)[1]  # skip 0