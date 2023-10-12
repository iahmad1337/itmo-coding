import utils
import pytest
import numpy as np
import numpy.typing as npt
import itertools

@pytest.mark.parametrize(
    "string,matrix", 
    [
        ("000", [[0, 0, 0]]), 
        ("111 111", [[1, 1, 1], [1, 1, 1]]),
        ("0\n0\n0", [[0], [0], [0]]),
        ("01\n 10 \n 01", [[0, 1], [1, 0], [0, 1]]),
    ]
)
def test_string_to_matrix(string: str, matrix: [[int]]) -> None:
    assert utils.string_to_matrix(string) == matrix

def generate_binary_matrix(n: int, m: int, seed: int = 1337) -> npt.ArrayLike:
    rs = np.random.RandomState(seed)
    return rs.choice([0, 1], size=(n, m))

@pytest.mark.parametrize(
    "n,m,seed",
    list(itertools.product(range(1, 10), range(1, 10), range(1, 10)))
)
def test_row_echelon_form(n: int, m: int, seed: int) -> None:
    matrix = generate_binary_matrix(n, m, seed)
    assert utils.is_row_echelon_form(utils.row_echelon_form(matrix))