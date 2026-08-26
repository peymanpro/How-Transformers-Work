from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Matrix:
    _data: np.ndarray

    def __post_init__(self) -> None:
        data = np.asarray(
            self._data,
            dtype=np.float64,
        )

        if data.ndim != 2:
            raise ValueError(
                "Matrix data must be two-dimensional."
            )

        if data.size == 0:
            raise ValueError(
                "Matrix cannot be empty."
            )

        if not np.isfinite(data).all():
            raise ValueError(
                "Matrix values must be finite."
            )

        object.__setattr__(
            self,
            "_data",
            data.copy(),
        )

    @classmethod
    def from_values(
        cls,
        values: list[list[float]],
    ) -> Matrix:
        return cls(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._data.shape)

    @property
    def rows(self) -> int:
        return int(self._data.shape[0])

    @property
    def columns(self) -> int:
        return int(self._data.shape[1])

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    def transpose(self) -> Matrix:
        return Matrix(self._data.T)

    def multiply(
        self,
        other: Matrix,
    ) -> Matrix:
        if self.columns != other.rows:
            raise ValueError(
                "Matrix dimensions are incompatible for multiplication: "
                f"{self.shape} × {other.shape}."
            )

        return Matrix(
            self._data @ other._data
        )

    def add(
        self,
        other: Matrix,
    ) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(
                "Matrix dimensions must match for addition: "
                f"{self.shape} != {other.shape}."
            )

        return Matrix(
            self._data + other._data
        )

    def multiply_scalar(
        self,
        scalar: float,
    ) -> Matrix:
        if not np.isfinite(scalar):
            raise ValueError(
                "Scalar must be finite."
            )

        return Matrix(
            self._data * scalar
        )
