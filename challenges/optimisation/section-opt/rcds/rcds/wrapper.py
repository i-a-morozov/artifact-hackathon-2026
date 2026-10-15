"""
Objective wrapper module.
Rescale input knobs, attach utility methods, retry, regularization, caching and logging.

"""
from __future__ import annotations

import numpy
import torch

from functools import wraps

from torch import Tensor
from typing import TypeAlias, Optional, Callable, Union

Knobs: TypeAlias = Tensor
Value: TypeAlias = Tensor
Error: TypeAlias = Tensor

Objective: TypeAlias = Callable[[Knobs], tuple[Value, Error]]

class Wrapper():
    """
    Objective function wrapper
    Can be used as a decorator for objective wrapping (rescale knobs, attach utility methods, retry, regularization, caching and logging)

    Note, if the original objective is differentible, so will be the wrapped objective (can fail with caching)

    Original objective is expected to return a (Value, Error) tuple of tensors (scalars)
    If sf parameter is True, sign of the original objective returned value will be flipped (minimization/maximization)
    If returned objective value is tensor(nan), upto additional nr evaluations will be performed (retry)
    The case when the final objective value is tensor(nan) is not handled
    If error parameter is None, objective returned error is used, else it is replaced by a given error parameter in all subsequent evaluations
    If regularization parameters l1 and/or l2 are not zero, regularization penalty is added to the returned objective value
    value -> value + l1*(knobs**1).abs().sum() + l2*(knobs**2).abs().sum()
    
    Given an original objective: Knobs -> (Value, Error), wrapped objective (knobs are in unit cube)

    @Wrapper(...)
    def objective(knobs: Knobs, ...) -> tuple[Value, Error]:
        ...

    Wrapped objective must be first 'evaluated' without parameters to attach attributes and methods
    Note, original objective is not evaluated in this case

    Significance steps parameter dk is not used in objective evaluation
    It is attached to wrapped objective in units of rescaled knobs (unit cube knobs)
    Parameter dk can be used during optimization to decide whether a particular knob is worthy to change
    For example, if suggested change for knob k is less than dk, old value can be kept
    This is useful if knob change itself is expensive

    Parameters
    ----------
    cache: bool, default=False
        flag to cache knobs (original box cube), value and error
    sf: bool, default=False
        flag to flip original objective value sign
    nr: int, non-negative, default=0
        maximum number of retrials if returned objective value is tensor(nan)
    nk: int, positive, default=1
        number of knobs
    lb: Optional[Union[Tensor, list[float]]]
        knobs lower bounds
    ub: Optional[Union[Tensor, list[float]]]
        knobs upper bounds
    dk: Optional[Union[Tensor, list[float]]]
        knobs significance steps
    error: Optional[float]
        objective error value, if not None, replace original objective returned error
    l1: Optional[float]
        l1 regularization factor
    l2: Optional[float]
        l2 regularization factor
    dtype: torch.dtype, default=torch.float64
        data type
    device: torch.device, default=torch.device('cpu')
        data device

    Attributes
    ----------
    cache: bool, default=False
        flag to cache knobs (original box cube), value and error
    sf: bool, default=False
        flag to flip original objective value sign
    nr: int, non-negative, default=0
        maximum number of retrials if returned objective value is tensor(nan)
    nk: int, positive, default=1
        number of knobs
    lb: Tensor
        knobs lower bounds
    ub: Tensor
        knobs upper bounds
    dk: Tensor
        knobs rescaled significance steps
    error: Tensor
        objective error value, if not None, replace original objective returned error
    l1: Tensor
        l1 regularization factor
    l2: Tensor
        l2 regularization factor
    dtype: torch.dtype, default=torch.float64
        data type
    device: torch.device, default=torch.device('cpu')
        data device
    nan: Tensor
        nan matched to dtype & device
    history_knobs: list[list[float]]
        cached knobs (original cube)
    history_value: list[float]
        cached objective values
    history_error: list[float]
        cached objective errors

    Methods
    ----------
    __init__(self, *, cache:bool=False, sf:bool=False, nr:int=0, nk:int=1, lb:Optional[Union[Tensor, list[float]]]=None, ub:Optional[Union[Tensor, list[float]]]=None, dk:Optional[Union[Tensor, list[float]]]=None, error:Optional[float]=None, l1:Optional[float]=None, l2:Optional[float]=None, dtype:torch.dtype=torch.float64, device:torch.device=torch.device('cpu')) -> None
        Objective function wrapper initialization.
    __repr__(self) -> str
        String representation.
    forward(self, knobs:Knobs) -> Knobs
        Rescale original input knobs into unit ouput knobs.
    inverse(self, knobs:Knobs) -> Knobs
        Rescale unit input knobs into original output knobs.
    history(self) -> tuple[Knobs, Value, Error]
        Return cached knobs, value and error data as a tuple of tensors.
    __call__(self, objective:Objetive, *args, wrapped:bool=True) -> Objetive
        Return wrapped objective.

    """
    def __init__(self,
                 *,
                 cache:bool=False,
                 sf:bool=False,
                 nr:int=0,
                 nk:int=1,
                 lb:Optional[Union[Tensor, list[float]]]=None,
                 ub:Optional[Union[Tensor, list[float]]]=None,
                 dk:Optional[Union[Tensor, list[float]]]=None,
                 error:Optional[float]=None,
                 l1:Optional[float]=None,
                 l2:Optional[float]=None,
                 dtype:torch.dtype=torch.float64,
                 device:torch.device=torch.device('cpu')) -> None:
        """
        Objective function wrapper initialization.

        Parameters
        ----------
        cache: bool, default=False
            flag to cache knobs (original box cube), value and error
        sf: bool, default=False
            flag to flip originalobjective value sign
        nr: int, non-negative, default=0
            maximum number of retrials if returned objective value is tensor(nan)
        nk: int, positive, default=1
            number of knobs
        lb: Optional[Union[Tensor, list[float]]]
            knobs lower bounds
        ub: Optional[Union[Tensor, list[float]]]
            knobs upper bounds
        dk: Optional[Union[Tensor, list[float]]]
            knobs significance steps
        error: Optional[float]
            objective error value, if not None, replace original objective returned error
        l1: Optional[float]
            l1 regularization factor
        l2: Optional[float]
            l2 regularization factor
        dtype: torch.dtype, default=torch.float64
            data type
        device: torch.device, default=torch.device('cpu')
            data device

        Returns
        -------
        None

        """
        self.dtype, self.device = dtype, device

        self.nan:Tensor = torch.tensor(float('nan'), dtype=self.dtype, device=self.device)

        self.history_knobs:list[list[float]]=[]
        self.history_value:list[float]=[]
        self.history_error:list[float]=[]

        self.cache:bool = cache
        if not isinstance(self.cache, bool):
            raise TypeError(f'WRAPPER: expected bool value for cache')

        self.sf:bool = sf
        if not isinstance(self.sf, bool):
            raise TypeError(f'WRAPPER: expected bool value for sf')

        self.nr:int = nr
        if not isinstance(self.nr, int):
            raise TypeError(f'WRAPPER: expected int value for nr')
        if self.nr < 0:
            raise ValueError(f'WRAPPER: expected nr >= 0')

        self.nk:int = nk
        if not isinstance(self.nk, int):
            raise TypeError(f'WRAPPER: expected int value for nk')
        if self.nk < 1:
            raise ValueError(f'WRAPPER: expected nk >= 1')

        self.lb:Tensor = lb
        if self.lb is not None:
            if len(self.lb) != self.nk:
                raise Exception(f'WRAPPER: lb size mismatch, expected length {self.nk}, got {len(self.lb)} on input')
            if not isinstance(self.lb, Tensor):
                try:
                    self.lb = torch.tensor(self.lb, dtype=self.dtype, device=self.device)
                except TypeError:
                    raise TypeError(f'WRAPPER: failed to convert lb data to torch')
            else:
                self.lb = self.lb.to(dtype=self.dtype, device=self.device)
        else:
            self.lb = torch.zeros(self.nk, dtype=self.dtype, device=self.device)

        self.ub:Tensor = ub
        if self.ub is not None:
            if len(self.ub) != self.nk:
                raise Exception(f'WRAPPER: ub size mismatch, expected length {self.nk}, got {len(self.ub)} on input')
            if not isinstance(self.ub, Tensor):
                try:
                    self.ub = torch.tensor(self.ub, dtype=self.dtype, device=self.device)
                except TypeError:
                    raise TypeError(f'WRAPPER: failed to convert ub data to torch')
            else:
                self.ub = self.ub.to(dtype=self.dtype, device=self.device)
        else:
            self.ub = torch.ones(self.nk, dtype=self.dtype, device=self.device)

        if (self.ub - self.lb <= 0).sum() != 0:
            raise Exception(f'WRAPPER: bounds mismatch')

        self.dk:Tensor = dk
        if self.dk is not None:
            if len(self.dk) != self.nk:
                raise Exception(f'WRAPPER: dk size mismatch, expected length {self.nk}, got {len(self.dk)} on input')
            if not isinstance(self.dk, Tensor):
                try:
                    self.dk = torch.tensor(self.dk, dtype=self.dtype, device=self.device)
                except TypeError:
                    raise Exception(f'WRAPPER: failed to convert dk data to torch')
            else:
                self.dk = self.dk.to(dtype=self.dtype, device=self.device)
        else:
            self.dk = torch.zeros(self.nk, dtype=self.dtype, device=self.device)

        if (self.ub - self.lb <= self.dk).sum() != 0:
            raise Exception(f'WRAPPER: dk values mismatch')

        self.dk /= self.ub - self.lb

        self.error:Tensor = error
        if self.error is not None:
            if not isinstance(self.error, float):
                raise TypeError(f'WRAPPER: expected float for error parameter')
            self.error = torch.tensor(self.error, dtype=self.dtype, device=self.device)

        self.l1:Tensor = l1
        if self.l1 is not None:
            if not isinstance(self.l1, float):
                raise TypeError(f'WRAPPER: expected float for l1 parameter')
            self.l1 = torch.tensor(self.l1, dtype=self.dtype, device=self.device)

        self.l2:Tensor = l2
        if self.l2 is not None:
            if not isinstance(self.l2, float):
                raise TypeError(f'WRAPPER: expected float for l2 parameter')
            self.l2 = torch.tensor(self.l2, dtype=self.dtype, device=self.device)


    def __repr__(self) -> str:
        """
        String representation.

        Note, data type and device are not included

        Parameters
        ----------
        None

        Returns
        ----------
        string representation (str)

        """
        return fr'Wrapper(cache={self.cache}, sf={self.sf}, nr={self.nr}, nk={self.nk}, lb={self.lb.tolist()}, ub={self.ub.tolist()}, dk={(self.dk*(self.ub - self.lb)).tolist()}, error={self.error if not self.error else self.error.item()}, l1={self.l1 if not self.l1 else self.l1.item()}, l2={self.l2 if not self.l2 else self.l2.item()})'


    def forward(self,
                knobs:Knobs) -> Knobs:
        """
        Rescale original input knobs into unit ouput knobs.

        Parameters
        ----------
        knobs: Knobs
            input knobs to rescale

        Returns
        -------
        rescaled output knobs (Knobs)

        """
        return (knobs - self.lb)/(self.ub - self.lb)


    def inverse(self,
                knobs:Knobs) -> Knobs:
        """
        Rescale unit input knobs into original output knobs.

        Parameters
        ----------
        knobs: Knobs
            input knobs to rescale

        Returns
        -------
        rescaled output knobs (Knobs)

        """
        return knobs*(self.ub - self.lb) + self.lb


    def history(self) -> tuple[Knobs, Value, Error]:
        """
        Return cached knobs, values and errors as a tuple of tensors.

        Parameters
        ----------
        None

        Returns
        -------
        knobs, values, errors (tuple[Knobs, Value, Error])

        """
        knobs = torch.tensor(self.history_knobs, dtype=self.dtype, device=self.device)
        value = torch.tensor(self.history_value, dtype=self.dtype, device=self.device)
        error = torch.tensor(self.history_error, dtype=self.dtype, device=self.device)

        return knobs, value, error


    def __call__(self,
                 objective:Objective,
                 *args,
                 wrapped:bool=True) -> Objective:
        """
        Return wrapped objective.

        Evaluate output without arguments to attach attributes and methods
        Evaluate output with input knobs (in unit cube) to return (value, error) tuple

        Parameters
        ----------
        objective: Objective
            original objective
        *args:
            passed to objective
        wrapped: bool, default=True
            flag to return wrapped objective

        Returns
        -------
        wrapped objective (Objective)

        """
        @wraps(objective)
        def wrapper(knobs:Optional[Knobs]=None, *args) -> tuple[Value, Error]:

            if knobs is None:
                wrapper.__dict__ = self.__dict__

                wrapper.forward = self.forward
                wrapper.inverse = self.inverse
                wrapper.history = self.history
                
                wrapper.history_knobs = []
                wrapper.history_value = []
                wrapper.history_error = []

                wrapper.counter:int = 0
                wrapper.n:int = 0
                wrapper.m:int = 0

            else:

                knobs = wrapper.inverse(knobs) if wrapped else knobs
                value = wrapper.nan

                wrapper.n += 1

                while torch.isnan(value):
                    wrapper.m += 1
                    value, error = objective(knobs, *args)
                    wrapper.counter += 1
                    if wrapper.counter > wrapper.nr: break

                wrapper.counter = 0

                if wrapper.error is not None:
                    error = wrapper.error

                if wrapper.l1 is not None: value += wrapper.l1*(knobs**1).abs().sum()
                if wrapper.l2 is not None: value += wrapper.l2*(knobs**2).abs().sum()

                value = value if not wrapper.sf else -value

                if wrapper.cache:
                    wrapper.history_knobs.append(knobs.detach().cpu().tolist())
                    wrapper.history_value.append(value.detach().cpu().tolist())
                    wrapper.history_error.append(error.detach().cpu().tolist())

                return value, error

        return wrapper


def main():
    pass

if __name__ == '__main__':
    main()