#!/usr/bin/env python3

import abc

class Model(metaclass=abc.ABCMeta):
    """Abstract base for models."""

    @abc.abstractmethod
    def n(self) -> int:
        """Returns the number of cycles in the model."""
        pass

    @abc.abstractmethod
    def run(cycles: int) -> int:
        """Advances the model CYCLES iterations, returning cycle count."""
        pass

    @abc.abstractmethod
    def dump(self) -> dict[str, tuple[float, ...]]:
        """Returns model state as a dictionary."""
        pass


def check_norm(tag: str, x: float) -> None:
    """Raises a ValueError if x is not in range 0 <= x <= 1"""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"invalid {tag}: {x}")

def clip(value: float) -> float:
    return min(1.0, max(0.0, value))
    
def step(values: list[float], delta: float) -> None:
    values.append(clip(values[-1] + delta))

class SIRModel(Model):
    """Basic SIR model."""

    def __init__(self, i0: float, alpha: float, beta: float):
        check_norm("i0", i0)
        check_norm("alpha", alpha)
        check_norm("beta", beta)
        self._alpha = alpha
        self._beta = beta
        self._i: list[float] = [i0]
        self._s: list[float] = [1.0 - i0]
        self._r: list[float] = [0.0]

    def n(self) -> int:
        return len(self._i)

    def run(self, cycles: int) -> int:
        if not cycles > 0:
            raise ValueError(f"invalid cycle count: {cycles}")
        for i in range(cycles):
            infections = self._alpha * self._i[-1] * self._s[-1]
            removals = self._beta * self._i[-1]
            step(self._s, -infections)
            step(self._i, infections - removals)
            step(self._r, removals)
        return self.n()

    def dump(self) -> dict[str, tuple[float, ...]]:
        return {
            "s": tuple(self._s),
            "i": tuple(self._i),
            "r": tuple(self._r),
        }

def lag(values: list[float], tau: int) -> float:
    if len(values) > tau:
        return values[-(tau + 1)]
    return 0.0

class SIRXModel(Model):
    """Model with permanent removal and re-susceptibility."""
    def __init__(self, i0: float, alpha: float, beta: float,
                 rho: float, delta: float, tau: int):
        check_norm("alpha", alpha)
        if beta + delta >= 1.0:
            raise ValueError(f"rho + delta too large: {rho + delta}")
        self._alpha = alpha
        self._beta = beta
        self._delta = delta
        self._rho = rho
        self._tau = tau
        self._i: list[float] = [i0]
        self._s: list[float] = [1.0 - i0]
        self._r: list[float] = [0.0]
        self._x: list[float] = [0.0]
        self._infections = []
        self._recoveries = []
        self._deaths = []
        self._lapses = []
        
    def n(self) -> int:
        return len(self._i)

    def run(self, cycles: int) -> int:
        if not cycles > 0:
            raise ValueError(f"invalid cycle count: {cycles}")
        for i in range(cycles):
            self._infections.append(self._alpha * self._i[-1] * self._s[-1])
            self._recoveries.append(self._beta * self._i[-1])
            self._deaths.append(self._delta * self._i[-1])
            self._lapses.append(self._rho * lag(self._recoveries, self._tau))
            step(self._s, self._lapses[-1] - self._infections[-1])
            step(self._i, self._infections[-1] - self._recoveries[-1] - self._deaths[-1])
            step(self._r, self._recoveries[-1] - self._lapses[-1])
            step(self._x, self._deaths[-1])
        return self.n()

    def dump(self) -> dict[str, tuple[float, ...]]:
        return {
            "s": tuple(self._s),
            "i": tuple(self._i),
            "r": tuple(self._r),
            "x": tuple(self._x),
        }
