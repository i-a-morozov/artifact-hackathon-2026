"""
Minimization module.
RCDS minimization (modified Powell's method for scalar noisy objective with line minimization by parabola fit or gp & bo).
Interface to scipy minimize, torch optimize.
Derivative based optimization (Adam & Newton).
Nonlinear regression, standard errors.

"""
from __future__ import annotations

import numpy
import torch

import logging
import warnings

from torch import Tensor
from typing import TypeAlias, Optional, Callable, Union

from pydantic.v1 import validate_arguments

class Config:
    arbitrary_types_allowed = True

Knobs: TypeAlias = Tensor
Value: TypeAlias = Tensor
Error: TypeAlias = Tensor

Objective: TypeAlias = Callable[[Knobs], tuple[Value, Error]]

from time import sleep

from statsmodels.api import WLS

from scipy.optimize import minimize

from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf

from .statistics import median, biweight_midvariance
from .wrapper import Wrapper

class Minimizer():
    """
    Minimizer class.
    RCDS single (noisy) objective minimization.
    Wrappers to scipy (bounded methods) and torch (SGD, Adam and other) optimization.
    Derivative based optimization (Adam & Newton).
    Nonlinear regression, standard errors.

    Parameters
    ----------
    objective: Objective
        (wrapped) objective
    epsilon: float, positive, default=1.0E-16
        float epsilon
    file: Optional[str]
        output file name (logging)

    Attributes
    ----------
    objective: Objective
        wrapped objective
    dtype: torch.dtype, default=torch.float64
        data type
    device: torch.device, default=torch.device('cpu')
        data device
    nan: Tensor
        nan matched to dtype & device
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
    golden: float
        golden ratio
    epsilon: Tensor
        float epsilon
    file: str
        output file name (logging)
    history_neval: int
        number of objective evaluations
    history_niter: int
        number of iterations
    history_knobs: list[list[float]]
        knobs history
    history_value:list[float]
        value history
    history_error:list[float]
        error history

    Methods
    ----------
    __init__(self, /, objective:Objective, *, epsilon:float=1.0E-16, file:Optional[str]=None) -> None
        Minimizer initialization.
    __repr__(self) -> str
        String representation.
    forward(self, knobs:Knobs) -> Knobs
        Rescale original input knobs into unit output knobs.
    inverse(self, knobs:Knobs) -> Knobs
        Rescale unit input knobs into original output knobs.
    history(self) -> tuple[Knobs, Value, Error]
        Return cached knobs, values and errors as a tuple of tensors.
    clean(self) -> None
        Clean history.
    append(self, knobs:Knobs, value:Value, error:Error) -> None
        Append input knobs, value and error to cache.
    close_knobs(self, present:Knobs, sequent:Knobs) -> bool
        Check whether knobs are close based on knobs significance defined by objective.
    alter_knobs(self, present:Knobs, sequent:Knobs) -> Knobs
        Alter knobs components based on knobs significance.
    termination_knobs(self, present:Knobs, sequent:Knobs, *, tolerance:Union[Tensor, float]=1.0E-9) -> bool
        Check termination condition for knobs.
    termination_value(self, present:tuple[Value, Error], sequent:tuple[Value, Error], *, tolerance:Union[Tensor, float]=1.0E-9) -> bool
        Check termination condition for objective value.
    interval(knobs:Knobs, vector:Tensor) -> tuple[Tensor, Tensor]
        Return interval of parameter values (min, max) for which all knobs remain in the unit cube for given initial knobs and direction.
    to_cube(knobs:Knobs) -> Knobs
        Move knobs to the unit cube.
    on_cube(self, knobs:Knobs) -> bool
        Check whether knobs are on the unit cube.
    adjust_cube(self, *, data:Opional[Tensor]=None, extend:bool=False, factor:float=5.0, center:Callable[[Tensor], Tensor]=median, spread:Callable[[Tensor], Tensor]=biweight_midvariance) -> tuple[Tensor, Tensor]
        Adjust search box using cached knobs (robust) dispersion.
    matrix(self, *args, knobs:Optional[Knobs]=None) -> Tensor
        Generate direction matrix.
    bracket(self, sf:Union[Tensor, float], knobs:Knobs, value:Value, error:Error, vector:Tensor, *args, sf_min:float=1.0E-3, sf_max:float=1.0E-1, fc:float=3.0) -> tuple[Tensor, Tensor, Tensor, Tensor]
        Bracket objective minimum for given initial knobs along a given search direction.
    parabola(self, vector:Tensor, table_alpha:Tensor, table_knobs:Tensor, table_value:Tensor, table_error:Tensor, *args, np:int=2, ns:int=4, fr:float=0.1, ft:float=5.0, tolerance:float=1.0E-6, sample:bool=True, detector:Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]=None) -> tuple[Tensor, Tensor, Tensor]
        Find 1D minimum using (weighted) parabola fit.
    minimize_parabola(self, sf:Union[Tensor, float], knobs:Knobs, value:Value, error:Error, vector:Tensor, *args, sf_min:float=1.0E-3, sf_max:float=1.0E-1, fc:float=3.0, np:int=2, ns:int=4, fr:float=0.1, ft:float=5.0, tolerance:float=1.0E-6, sample:bool=True, detector:Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]=None) -> tuple[Knobs, Value, Error]
        Parabola line minimization (bracket and 1D minimization using parabola fit).
    minimize_gp(self, sf:Union[Tensor, float], knobs:Knobs, value:Value, error:Error, vector:Tensor, *args, no_ucb:int=4, no_ei:int=4, nr:int=64, rs:int=256, beta:Union[Tensor, float]=25.0, use_parabola:bool=True, np:int=1) -> tuple[Knobs, Value, Error]
        GP line minimization (GP and optional 1D parabola fit).
    rcds(self, knobs:Knobs, matrix:Tensor, *args, ni:int=8, sf:float=0.01, sf_min:float=0.001, sf_max:float=0.1, dr:float=1.0, otol:float=1.0E-6, ktol:float=1.0E-6, minimize:Optional[Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]]=None, factor:float=0.0, termination:bool=True, verbose:bool=False, pause:float=0.0, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]
        RCDS minimization.
    scipy(self, knobs:Knobs, *args, method:str='powell', **kwargs)
        Scipy minimization wrapper for selected bounded methods.
    torch(self, knobs: Knobs, count:int, optimizer, *args, **kwargs) -> tuple[Tensor, Tensor]
        Torch SGD (or other optimizer) minimization.
    newton(self, knobs:Tensor, jacobian:Callable, *args, count:int=1, factor:float=1.0, alpha:float=0.0) -> tuple[Tensor, Tensor]
        Newton minimization.
    adam(self, knobs:Tensor, jacobian:Callable, *args, count:int=1, lr:float=0.01, betas:tuple[float, float]=(0.900, 0.999), epsilon:float=1.0E-9) -> tuple[Tensor, Tensor]
        Adam minimization.
    standard_errors(knobs:Knobs, x:Tensor, y:Tensor, model:Callable[[Tensor], Tensor], objective:Callable[[Tensor], Tensor], estimator:Callable[[Tensor], Tensor], *args) -> Tensor
        Estimate standard errors.

    """
    @validate_arguments(config=Config)
    def __init__(self,
                 /,
                 objective:Objective,
                 *,
                 epsilon:float=1.0E-16,
                 file:Optional[str]=None) -> None:

        """
        Minimizer initialization.

        Note, wrapped objective is assumed to be 'pre-evaluated'

        Parameters
        ----------
        objective: Objective
            objective
        epsilon: float, positive, default=1.0E-16
            float epsilon
        file: Optional[str]
            output file name

        Returns
        -------
        None

        """
        self.objective:Objective = objective

        self.dtype:torch.dtype = self.objective.dtype
        self.device:torch.device = self.objective.device

        self.nan:Tensor = self.objective.nan

        self.nk:int = self.objective.nk
        
        self.lb:Tensor = self.objective.lb
        self.ub:Tensor = self.objective.ub
        self.dk:Tensor = self.objective.dk

        self.error:Tensor = self.objective.error

        self.l1:Tensor = self.objective.l1
        self.l2:Tensor = self.objective.l2

        self.forward:Callable[[Knobs], Knobs] = self.objective.forward
        self.inverse:Callable[[Knobs], Knobs] = self.objective.inverse

        self.golden:float = 0.5*(1.0 + 5.0**0.5)

        self.epsilon:float = epsilon
        if not isinstance(self.epsilon, float):
            raise TypeError(f'MINIMIZER: expected float value for epsilon')
        if self.epsilon <= 0.0:
            raise ValueError(f'MINIMIZER: expected epsilon > 0.0')
        self.epsilon:Tensor = torch.tensor(self.epsilon, dtype=self.dtype, device=self.device)

        self.file:str = file
        if not isinstance(self.file, str) and self.file is not None:
            raise TypeError(f'MINIMIZER: expected file value for file')
        if self.file:
            logging.basicConfig(filename=self.file, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

        self.history_neval:int = 0
        self.history_niter:int = 0
        self.history_knobs:list[list[float]] = []
        self.history_value:list[float] = []
        self.history_error:list[float] = []


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
        return fr'Minimizer({self.objective}, epsilon={self.epsilon.item()}, file={self.file})'


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


    def clean(self) -> None:
        """
        Clean history.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.history_neval:int = 0
        self.history_niter:int = 0
        self.history_knobs:list[list[float]] = []
        self.history_value:list[float] = []
        self.history_error:list[float] = []


    def append(self,
               knobs:Knobs,
               value:Value,
               error:Error) -> None:
        """
        Append input knobs, value and error to cache.

        Note, increments objective evaluation counter, optional logging

        Parameters
        ----------
        knobs, value, error: Knobs, Value, Error
            input knobs, value and error

        Returns
        -------
        None

        """
        self.history_neval += 1

        if self.file:
            logging.info(''.join(f'{knob:>.9E} ' for knob in knobs) + f'{value:>.9E} ' + f'{error:>.9E} ')

        self.history_knobs.append(knobs.detach().cpu().tolist())
        self.history_value.append(value.detach().cpu().tolist())
        self.history_error.append(error.detach().cpu().tolist())


    def close_knobs(self,
                    present:Knobs,
                    sequent:Knobs) -> bool:
        """
        Check whether knobs are close based on knobs significance defined by objective.

        Parameters
        ----------
        present, sequent: Knobs
            knobs to compare

        Returns
        -------
        check result (bool)

        """
        return all((present - sequent).abs() <= self.epsilon + self.dk)


    def alter_knobs(self,
                    present:Knobs,
                    sequent:Knobs) -> Knobs:
        """
        Alter knobs components based on knobs significance.

        Parameters
        ----------
        present: Knobs
            present knobs
        sequent: Knobs
            sequent knobs

        Returns
        -------
        altered knobs (Knobs)

        """
        mask = (present - sequent).abs() <= self.epsilon + self.dk
        sequent[mask] = present[mask]
        return sequent


    def termination_knobs(self,
                          present:Knobs,
                          sequent:Knobs,
                          *,
                          tolerance:Union[Tensor, float]=1.0E-9) -> bool:
        """
        Check termination condition for knobs.
        
        Returns true if the difference norm between present and sequent is less than specified tolerance
        Returns true if present and sequent are close based on self.dk

        Parameters
        ----------
        present, sequent: Knobs
            knobs to compare
        tolerance: Union[Tensor, float], positive, default=1.0E-9
            termination tolerance

        Returns
        -------
        termination test result (bool)

        """
        return ((present - sequent).norm() <= self.epsilon + tolerance).cpu().item() or self.close_knobs(present, sequent)


    def termination_value(self,
                          present:tuple[Value, Error],
                          sequent:tuple[Value, Error],
                          *,
                          tolerance:Union[Tensor, float]=1.0E-9) -> bool:
        """
        Check termination condition for objective value.

        Parameters
        ----------
        present, sequent: tuple[Value, Error]
            values & errors to compare
        tolerance: Union[Tensor, float], positive, default=1.0E-9
            termination tolerance

        Returns
        -------
        termination test result (bool)

        """
        present_value, present_error = present
        sequent_value, sequent_error = sequent
        return ((present_value.abs() - sequent_value.abs()).abs() <= self.epsilon + tolerance + (present_error**2 + sequent_error**2).sqrt()).cpu().item()


    @staticmethod
    def interval(knobs:Knobs,
                 vector:Tensor) -> tuple[Tensor, Tensor]:
        """
        Return interval of parameter values (min, max) for which all knobs remain in the unit cube for given initial knobs and direction.

        Note, knobs += vector * parameter are in [0, 1] if parameter is in [min, max]
        
        Parameters
        ----------
        knobs: Knobs
            initial knobs (unit cube)
        vector: Tensor
            direction vector

        Returns
        -------
        valid parameter interval (tuple[Tensor, Tensor])

        """
        if knobs.shape != vector.shape or not torch.isfinite(knobs).all() or not torch.isfinite(vector).all():
            raise ValueError("Line-search point and direction must be finite and have matching shapes")
        moving = vector != 0
        if not moving.any():
            zero = knobs.new_tensor(0.0)
            return zero, zero
        bounds = torch.stack([
            -knobs[moving] / vector[moving],
            (1.0 - knobs[moving]) / vector[moving],
        ]).T.sort().values
        return bounds[:, 0].max(), bounds[:, 1].min()


    @staticmethod
    def to_cube(knobs:Knobs) -> Knobs:
        """
        Move knobs to the unit cube.

        Parameters
        ----------
        knobs: Knobs
            input knobs

        Returns
        -------
        changed knobs (Knobs)

        """
        knobs[knobs < 0.0] = 0.0
        knobs[knobs > 1.0] = 1.0
        return knobs


    def on_cube(self,
                knobs:Knobs) -> bool:
        """
        Check whether knobs are on the unit cube.

        Parameters
        ----------
        knobs: Knobs
            input knobs

        Returns
        -------
        check result (bool)

        """
        return any((knobs - 0.0).abs() < self.epsilon) or any((knobs - 1.0).abs() < self.epsilon)


    @validate_arguments(config=Config)
    def adjust_cube(self,
                    *,
                    data:Optional[Tensor]=None,
                    extend:bool=False,
                    factor:float=5.0,
                    center:Callable[[Tensor], Tensor]=median,
                    spread:Callable[[Tensor], Tensor]=biweight_midvariance) -> tuple[Tensor, Tensor]:
        """
        Adjust search box using cached knobs (robust) dispersion.

        Note, returned bounds are in original cube, use to redefine objective with new values

        Parameters
        ----------
        data: Optional[Tensor]
            knobs data (original cube)
        extend: bool
            flag to allow cube size increase
        factor: bool
            spread multiplication factor
        center: Callable[[Tensor], Tensor]
            center estimator
        spread: Callable[[Tensor], Tensor]
            spread estimator

        Returns
        -------
        lb, ub (tuple[Tensor, Tensor])

        """
        warnings.filterwarnings('ignore', category=UserWarning)

        if data is None:
            data, *_ = self.history()
            data = self.inverse(data)

        if not isinstance(data, Tensor):
            try:
                data = torch.tensor(data, dtype=self.dtype, device=self.device)
            except TypeError:
                raise TypeError(f'MINIMIZER: failed to convert knobs data to torch')

        data = data.T
        
        center = torch.func.vmap(center)(data) 
        spread = torch.func.vmap(spread)(data).sqrt()

        lb = center - factor*(self.epsilon + spread)
        ub = center + factor*(self.epsilon + spread)

        if not extend:
            lb[lb < self.lb] = self.lb[lb < self.lb]
            ub[ub > self.ub] = self.ub[ub > self.ub]

        return lb, ub


    @validate_arguments(config=Config)
    def matrix(self,
               *args,
               knobs:Optional[Knobs]=None) -> Tensor:
        """
        Generate direction matrix.

        Note, if knobs is not None, eigenvectors of objective hessian are computed


        Parameters
        ----------
        *args:
            passed to objective
        knobs: Optional[Knobs]
            evaluation point

        Returns
        -------
        direction matrix (Tensor)

        """
        if knobs is None:
            u, _, vh = torch.linalg.svd(torch.randn((self.nk, self.nk), dtype=self.dtype, device=self.device))
            return u @ vh

        cache, self.objective.cache = self.objective.cache, False
        try:
            hessian, _ = torch.func.hessian(self.objective)(knobs, *args)
        finally:
            self.objective.cache = cache

        # eigh returns eigenvectors as columns; rcds consumes matrix[i] as a row.
        hessian = 0.5 * (hessian + hessian.T)
        return torch.linalg.eigh(hessian).eigenvectors.T


    @validate_arguments(config=Config)
    def bracket(self,
                sf:Union[Tensor, float],
                knobs:Knobs,
                value:Value,
                error:Error,
                vector:Tensor,
                *args,
                sf_min:float=1.0E-3,
                sf_max:float=1.0E-1,
                fc:float=3.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Bracket objective minimum for given initial knobs along a given search direction.

        Parameters
        ----------
        sf: Union[Tensor, float], positive
            initial step fraction
        knobs: Knobs
            initial knobs
        value: Value
            initial objective value
        error: Error
            initial objective error
        vector: Tensor
            search direction vector
        *args:
            passed to objective
        sf_min: float, positive, default=1.0E-3
            min step fraction
        sf_max: float, positive, default=0.1
            max step fraction
        fc: float, positive, default=3.0
            factor to use in comparison of objective values

        Returns
        -------
        table_alpha, table_knobs, table_value, table_error (tuple[Tensor, Tensor, Tensor, Tensor])

        """
        alpha_lb, alpha_ub = self.interval(knobs, vector)

        step_min = sf_min*(alpha_ub - alpha_lb)
        step_max = sf_max*(alpha_ub - alpha_lb)

        step = sf*(alpha_ub - alpha_lb)

        table_alpha, table_knobs, table_value, table_error = [], [], [], []

        table_alpha.append(torch.zeros_like(step))
        table_knobs.append(knobs)
        table_value.append(value)
        table_error.append(error)

        alpha_min = torch.clone(torch.zeros_like(step))
        knobs_min = torch.clone(knobs)
        value_min = torch.clone(value)
        error_min = torch.clone(error)

        if step >= step_max: step = step_max
        if step <= step_min: step = step_min

        knobs_pos = self.to_cube(knobs + vector*step)
        knobs_pos = self.alter_knobs(knobs, knobs_pos)
        value_pos, error_pos = self.objective(knobs_pos, *args)
        self.append(knobs_pos, value_pos, error_pos)

        table_alpha.append(torch.clone(step))
        table_knobs.append(torch.clone(knobs_pos))
        table_value.append(torch.clone(value_pos))
        table_error.append(torch.clone(error_pos))

        if torch.isnan(value_pos):
            table_alpha = torch.stack(table_alpha) - alpha_min
            table_knobs = torch.stack(table_knobs)
            table_value = torch.stack(table_value)
            table_error = torch.stack(table_error)
            return table_alpha, table_knobs, table_value, table_error

        if value_pos - value_min < self.epsilon + (error_pos**2 + error_min**2).sqrt():
            alpha_min = torch.clone(step)
            knobs_min = torch.clone(knobs_pos)
            value_min = torch.clone(value_pos)
            error_min = torch.clone(error_pos)

        while value_pos - value_min <  self.epsilon + fc*(error_pos**2 + error_min**2).sqrt():

            if step == alpha_ub:
                break

            if step < step_max:
                step *= 1.0 + self.golden
            else:
                step += step_max

            if step > alpha_ub:
                step = torch.clone(alpha_ub)

            knobs_pos = self.to_cube(knobs + vector*step)
            knobs_pos = self.alter_knobs(knobs, knobs_pos)
            value_pos, error_pos = self.objective(knobs_pos, *args)
            self.append(knobs_pos, value_pos, error_pos)
            table_alpha.append(torch.clone(step))

            table_knobs.append(torch.clone(knobs_pos))
            table_value.append(torch.clone(value_pos))
            table_error.append(torch.clone(error_pos))

            if torch.isnan(value_pos):
                table_alpha = torch.stack(table_alpha) - alpha_min
                table_knobs = torch.stack(table_knobs)
                table_value = torch.stack(table_value)
                table_error = torch.stack(table_error)
                return table_alpha, table_knobs, table_value, table_error

            if value_pos - value_min < self.epsilon + (error_pos**2 + error_min**2).sqrt():
                alpha_min = torch.clone(step)
                knobs_min = torch.clone(knobs_pos)
                value_min = torch.clone(value_pos)
                error_min = torch.clone(error_pos)

        if value - value_min > self.epsilon + fc*(error**2 + error_min**2).sqrt():
            table_alpha = torch.stack(table_alpha) - alpha_min
            table_knobs = torch.stack(table_knobs)
            table_value = torch.stack(table_value)
            table_error = torch.stack(table_error)
            return table_alpha, table_knobs, table_value, table_error

        step = -sf*(alpha_ub - alpha_lb)

        if step.abs() >= step_max:
            step = -step_max
        if step.abs() <= step_min:
            step = -step_min

        knobs_neg = self.to_cube(knobs + vector*step)
        knobs_neg = self.alter_knobs(knobs, knobs_neg)
        value_neg, error_neg = self.objective(knobs_neg, *args)
        self.append(knobs_neg, value_neg, error_neg)

        table_alpha.append(torch.clone(step))
        table_knobs.append(torch.clone(knobs_neg))
        table_value.append(torch.clone(value_neg))
        table_error.append(torch.clone(error_neg))

        if torch.isnan(value_neg):
            table_alpha = torch.stack(table_alpha) - alpha_min
            table_knobs = torch.stack(table_knobs)
            table_value = torch.stack(table_value)
            table_error = torch.stack(table_error)
            return table_alpha, table_knobs, table_value, table_error

        if value_neg - value_min < self.epsilon + (error_neg**2 + error_min**2).sqrt():
            alpha_min = torch.clone(step)
            knobs_min = torch.clone(knobs_neg)
            value_min = torch.clone(value_neg)
            error_min = torch.clone(error_neg)

        while value_neg - value_min <  self.epsilon + fc*(error_neg**2 + error_min**2).sqrt():

            if step == alpha_lb:
                break

            if step.abs() < step_max:
                step *= 1.0 + self.golden
            else:
                step -= step_max

            if step < alpha_lb:
                step = torch.clone(alpha_lb)

            knobs_neg = self.to_cube(knobs + vector*step)
            knobs_neg = self.alter_knobs(knobs, knobs_neg)
            value_neg, error_neg = self.objective(knobs_neg, *args)
            self.append(knobs_neg, value_neg, error_neg)

            table_alpha.append(torch.clone(step))
            table_knobs.append(torch.clone(knobs_neg))
            table_value.append(torch.clone(value_neg))
            table_error.append(torch.clone(error_neg))

            if torch.isnan(value_neg):
                table_alpha = torch.stack(table_alpha) - alpha_min
                table_knobs = torch.stack(table_knobs)
                table_value = torch.stack(table_value)
                table_error = torch.stack(table_error)
                return table_alpha, table_knobs, table_value, table_error

            if value_neg - value_min < self.epsilon + (error_neg**2 + error_min**2).sqrt():
                alpha_min = torch.clone(step)
                knobs_min = torch.clone(knobs_neg)
                value_min = torch.clone(value_neg)
                error_min = torch.clone(error_neg)

        table_alpha = torch.stack(table_alpha) - alpha_min
        table_knobs = torch.stack(table_knobs)
        table_value = torch.stack(table_value)
        table_error = torch.stack(table_error)

        index = table_alpha.argsort()

        return table_alpha[index], table_knobs[index], table_value[index], table_error[index]


    @validate_arguments(config=Config)
    def parabola(self,
                 vector:Tensor,
                 table_alpha:Tensor,
                 table_knobs:Tensor,
                 table_value:Tensor,
                 table_error:Tensor,
                 *args,
                 np:int=2,
                 ns:int=4,
                 fr:float=0.1,
                 ft:float=5.0,
                 tolerance:float=1.0E-6,
                 sample:bool=True,
                 detector:Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]=None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Find 1D minimum using (weighted) parabola fit.

        If detector is not None, detector function is applied to fit residuals and is assumed to return a mask
        This mask is used to alter weights and refit parabola

        Parameters
        ----------
        vector: Tensor
            search direction
        table_alpha: Tensor
            alpha table
        table_knobs: Tensor
            knobs table
        table_value: Tensor
            value table
        table_error: Tensor
            error table
        *args:
            passed to objective
        np: int, positive, default=2
            number of points around current bracketed minimum (on each side) to use for parabola fit
        ns: int, positive, default=4
            number of points to add with uniform sampling in bracketed interval (defined by np)
        fr: float, positive, default=0.1
            factor to use in comparison of close points
        ft: float, positive, default=5.0
            threshold factor to for oulier detection
        tolerance: float, positive, default=1.0E-6
            objective tolerance
        sample: bool, default=True
            flag to add additional samples
        detector: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]
            detector function (residual, ft, otol)
            function to use for outlier cleaning, applied to fit residuals, assumed to return weights for each sample point

        Returns
        -------
        knobs, value, error (tuple[Tensor, Tensor, Tensor])

        """
        index = table_value.argmin()
        start = table_knobs[index]

        table_alpha = table_alpha[max(0, index - np) : index + np + 1]
        table_knobs = table_knobs[max(0, index - np) : index + np + 1]
        table_value = table_value[max(0, index - np) : index + np + 1]
        table_error = table_error[max(0, index - np) : index + np + 1]

        alpha_lb, *_, alpha_ub = table_alpha

        if sample:

            alpha_size = fr*(alpha_ub - alpha_lb)
            alpha_grid = torch.linspace(alpha_lb, alpha_ub, ns + 2, dtype=self.dtype, device=self.device)[1:-1]
            alpha_mask = torch.stack([(table_alpha - alpha).abs().min() > alpha_size for alpha in alpha_grid])
            alpha_grid = alpha_grid[alpha_mask]

            for alpha in alpha_grid:

                knobs = self.to_cube(start + vector*alpha)
                knobs = self.alter_knobs(start, knobs)
                value, error = self.objective(knobs, *args)
                self.append(knobs, value, error)

                table_alpha = torch.cat((table_alpha, alpha.unsqueeze(0)))
                table_knobs = torch.cat((table_knobs, knobs.unsqueeze(0)))
                table_value = torch.cat((table_value, value.unsqueeze(0)))
                table_error = torch.cat((table_error, error.unsqueeze(0)))

                if torch.isnan(value):
                    return knobs, value, error

        X = torch.stack([table_alpha**2, table_alpha, torch.ones_like(table_alpha)]).T.cpu().numpy()
        y = table_value.cpu().numpy()
        w = (1.0/table_error**2).nan_to_num(posinf=1.0)
        w = (w/w.sum()).cpu().numpy()
        fit = WLS(y, X, weights=w).fit()
        a, b, c = fit.params

        if detector is not None:

            residual = torch.tensor(fit.resid, dtype=self.dtype, device=self.device)
            mask = detector(residual, ft, tolerance)

            if mask.sum() == 0:
                mask = mask.logical_not()

            w = (mask/table_error**2).nan_to_num(posinf=1.0)
            w = (w/w.sum()).cpu().numpy()
            fit = WLS(y, X, weights=w).fit()
            a, b, c = fit.params

        if abs(a) >= self.epsilon:
            alpha = torch.tensor(-b/(2.0*a), dtype=self.dtype, device=self.device)
        else:
            index = table_value.argmin()
            alpha = table_alpha[index]

        if alpha > alpha_ub: alpha = alpha_ub
        if alpha < alpha_lb: alpha = alpha_lb

        knobs = self.to_cube(start + vector*alpha)
        knobs = self.alter_knobs(start, knobs)
        value, error = self.objective(knobs, *args)
        self.append(knobs, value, error)

        return knobs, value, error


    @validate_arguments(config=Config)
    def minimize_parabola(self,
                          sf:Union[Tensor, float],
                          knobs:Knobs,
                          value:Value,
                          error:Error,
                          vector:Tensor,
                          *args,
                          sf_min:float=1.0E-3,
                          sf_max:float=1.0E-1,
                          fc:float=3.0,
                          np:int=2,
                          ns:int=4,
                          fr:float=0.1,
                          ft:float=5.0,
                          tolerance:float=1.0E-6,
                          sample:bool=True,
                          detector:Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]=None) -> tuple[Knobs, Value, Error]:
        """
        Parabola line minimization (bracket and 1d minimization using parabola fit).

        Parameters
        ----------
        sf: Union[Tensor, float], positive
            initial step fraction
        knobs: Knobs
            initial knobs
        value: Value
            initial value
        error: Error
            initial error
        vector: Tensor
            search direction
        *args:
            passed to objective
        sf_min: float, positive, default=1.0E-3
            min step fraction
        sf_max: float, positive, default=0.1
            max step fraction
        fc: float, positive, default=3.0
            factor to use in comparison of objective values
        np: int, positive, default=2
            number of points around current bracketed minimum (on each side) to use for parabola fit
        ns: int, positive, default=4
            number of points to add with uniform sampling in bracketed interval (defined by np)
        fr: float, positive, default=0.1
            factor to use in comparison of close points
        ft: float, positive, default=5.0
            threshold factor to for oulier detection
        tolerance: float, positive, default=1.0E-6
            objective tolerance
        sample: bool, default=True
            flag to add additional samples
        detector: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]]
            detector function (residual, ft, otol)
            function to use for outlier cleaning, applied to fit residuals, assumed to return weights for each sample point

        Returns
        -------
        knobs, value, error (tuple[Knobs, Value, Error]])

        """
        alpha_lb, alpha_ub = self.interval(knobs, vector)
        if alpha_ub - alpha_lb <= max(self.epsilon, torch.finfo(knobs.dtype).eps):
            # A mixed-sign direction at a box corner may admit only alpha=0.
            return knobs.clone(), value.clone(), error.clone()
        alpha, knobs, value, error = self.bracket(sf, knobs, value, error, vector, *args, sf_min=sf_min, sf_max=sf_max, fc=fc)
        return self.parabola(vector, alpha, knobs, value, error, *args, np=np, ns=ns, fr=fr, ft=ft, tolerance=tolerance, sample=sample, detector=detector)


    @validate_arguments(config=Config)
    def minimize_gp(self,
                    sf:Union[Tensor, float],
                    knobs:Knobs,
                    value:Value,
                    error:Error,
                    vector:Tensor,
                    *args,
                    no_ucb:int=4,
                    no_ei:int=4,
                    ei_first:bool=False,
                    nr:int=64,
                    rs:int=256,
                    beta:Union[Tensor, float]=25.0,
                    use_parabola:bool=True,
                    np:int=1) -> tuple[Knobs, Value, Error]:
        """
        GP line minimization (GP and optional 1D parabola fit).

        Parameters
        ----------
        sf: Union[Tensor, float]
            initial step fraction (not used)
        knobs: Tensor
            initial knobs
        value: Tensor
            initial value
        error: Tensor
            initial error
        vector: Tensor
            search direction
        *args:
            passed to objective
        no_ucb: int, positive, default=4
            number of observations to perform with ucb af
        no_ei: int, positive, default=4
            number of observations to perform with ei af
        ei_first: bool, default=False
            use EI before UCB within each line search (otherwise UCB before EI)
        nr: int, positive, default=64
            number of restarts
        rs: int, positive, default=256
            number of raw samples
        beta: Union[Tensor, float], default=25.0
            ucb beta factor
        use_parabola: bool
            flag to perfrom parabola fit
        np: int, positive, default=1
            number of points to use in parabola fit

        Returns
        -------
        knobs, value, error (tuple[Knobs, Value, Error])

        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        alpha_lb, alpha_ub = self.interval(knobs, vector)
        if alpha_ub - alpha_lb <= max(self.epsilon, torch.finfo(knobs.dtype).eps):
            # No feasible move: skip this direction without evaluating the machine.
            return knobs.clone(), value.clone(), error.clone()

        @Wrapper(nk=1, lb=alpha_lb.flatten(), ub=alpha_ub.flatten(), dtype=self.dtype, device=self.device)
        def objective(alpha:Tensor, *args) -> tuple[Tensor, Tensor]:
            return self.objective(knobs + alpha*vector, *args)

        objective()

        beta = torch.as_tensor(beta, dtype=self.dtype, device=self.device).flatten()
        if beta.numel() == 1:
            beta = beta.expand(no_ucb)
        if beta.numel() != no_ucb or not torch.isfinite(beta).all() or (beta < 0).any():
            raise ValueError("beta must be nonnegative and scalar or one value per UCB step")

        x = torch.zeros((1, 1), dtype=self.dtype, device=self.device)
        X = torch.stack([objective.forward(alpha) for alpha in x])

        y = torch.clone(value).flatten()
        s = torch.clone(error).flatten()

        scale = (y.std() if y.numel() > 1 else y.new_tensor(1.0)).clamp_min(1.0E-8)
        Y = (y - y.mean())/scale
        S = s/scale

        gp = SingleTaskGP(X, Y.reshape(-1, 1), S.reshape(-1, 1).square().clamp_min(1.0E-8), outcome_transform=None)
        ll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(ll)

        bounds = torch.tensor([[0.0], [1.0]], dtype=self.dtype, device=self.device)

        for i in range(no_ei + no_ucb):
            best = torch.min(Y)
            use_ei = i < no_ei if ei_first else i >= no_ucb
            # LogEI has the same maximizer as EI and is numerically more stable.
            if use_ei:
                af = LogExpectedImprovement(gp, best, maximize=False)
            else:
                ucb_index = i - no_ei if ei_first else i
                af = UpperConfidenceBound(gp, beta[ucb_index], maximize=False)
            candidate, _ = optimize_acqf(af, bounds=bounds, num_restarts=nr, q=1, raw_samples=rs)
            candidate = candidate.flatten()
            value, error = objective(candidate, *args)
            x = torch.cat([x, objective.inverse(candidate).reshape(-1, 1)])
            X = torch.cat([X, candidate.reshape(-1, 1)])
            y = torch.cat([y, value.flatten()])
            s = torch.cat([s, error.flatten()])
            self.append(knobs + x[-1]*vector, y[-1], s[-1])
            scale = (y.std() if y.numel() > 1 else y.new_tensor(1.0)).clamp_min(1.0E-8)
            Y = (y - y.mean())/scale
            S = s/scale
            gp = SingleTaskGP(X, Y.reshape(-1, 1), S.reshape(-1, 1).square().clamp_min(1.0E-8), outcome_transform=None)
            ll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(ll)

        alpha = x.flatten()
        knobs = torch.stack([knobs + a*vector for a in alpha])
        value = y
        error = s

        index = alpha.argsort()
        alpha = alpha[index] - alpha[value.argmin()]
        knobs = knobs[index]
        value = value[index]
        error = error[index]

        index = value.argmin()

        if use_parabola:
            index = range(index - np, index + np + 1)
            return self.parabola(vector, alpha[index], knobs[index], value[index], error[index], *args, np=np, sample=False)

        return knobs[index], value[index], error[index]


    @validate_arguments(config=Config)
    def rcds(self,
             knobs:Knobs,
             matrix:Tensor,
             *args,
             ni:int=8,
             sf:float=0.01,
             sf_min:float=0.001,
             sf_max:float=0.1,
             dr:float=1.0,
             otol:float=1.0E-6,
             ktol:float=1.0E-6,
             minimize:Optional[Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]]=None,
             factor:float=0.0,
             termination:bool=True,
             verbose:bool=False,
             warning:bool=True,
             pause:float=0.0,
             **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        RCDS minimization.

        Parameters
        ----------
        knobs: Knobs
            initial knobs (unit cube)
        matrix: Tensor
            initial search directions
        *args:
            passed to objective
        ni: int, positive, default=8
            max number of iterations
        sf: float, positive, default=0.01
            initial step fraction
        sf_min: float, positive, default=0.001
            min step fraction
        sf_max: float, positive, default=0.1
            max step fraction
        dr: float, positive, default=1.0
            step fraction decay rate (applied after each iteration)
        otol: float, positive, default=1.0E-6
            objective tolerance
        ktol: float, positive, default=1.0E-6
            knobs tolerance
        minimize: Optional[Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]]
            minimizer function (minimize_parabola or minimize_gp)
        factor: float, default=0.0
            proximity factor in [0.0, 1.0]
        termination: bool, default=True
            flag to use knobs and value termination
        verbose: bool, default=False
            verbose flag
        warning: bool, default=True
            warning flag
        pause: float, default=0.0
            pause after each iteration in seconds
        kwargs:
            minimize specific options

        Returns
        -------
        knobs, value, error, matrix (tuple[Tensor, Tensor, Tensor, Tensor])

        """
        minimize = self.minimize_parabola if minimize is None else minimize

        sf = torch.tensor(sf, dtype=self.dtype, device=self.device)

        knobs = torch.clone(knobs)
        if not torch.allclose(knobs, self.to_cube(torch.clone(knobs))):
            if warning:
                print(f'WARNING: initial knobs are not in the unit cube, values are adjusted')
            knobs = self.to_cube(knobs)

        matrix = torch.clone(matrix)
        if not torch.linalg.matrix_rank(matrix) == self.nk:
            if warning:
                print(f'WARNING: initial matrix is not full rank, changed to identity matrix')
            matrix = torch.eye(self.nk, dtype=self.dtype, device=self.device)

        k = 0

        xk = torch.zeros((ni + 1, *knobs.shape), dtype=self.dtype, device=self.device)
        fk = torch.zeros(ni + 1, dtype=self.dtype, device=self.device)
        sk = torch.zeros(ni + 1, dtype=self.dtype, device=self.device)
        mk = torch.zeros((ni + 1, *matrix.shape), dtype=self.dtype, device=self.device)

        xk[k] = torch.clone(knobs)
        fk[k], sk[k] = self.objective(xk[k], *args)
        mk[k] = torch.clone(matrix)

        self.append(xk[k], fk[k], sk[k])

        if torch.isnan(fk[k]):
            print(f'EXIT: objective value is nan')
            return sf, xk[:k+1], fk[:k+1], sk[:k+1], mk[:k+1]

        if verbose:
            print(f'-----------------------------')
            print(f'Iteration {k}:')
            print(f'-----------------------------')
            print(f'delta: {sf.cpu().numpy()}')
            print(f'knobs: {self.objective.inverse(xk[k]).cpu().numpy()}')
            print(f'value: {fk[k].cpu().numpy()}')
            print(f'error: {sk[k].cpu().numpy()}')
            print(f'count: {self.history_neval}')
            print()

        i = 0
        n = self.nk

        xi = torch.zeros((n + 1, *knobs.shape), dtype=self.dtype, device=self.device)
        fi = torch.zeros(n + 1, dtype=self.dtype, device=self.device)
        si = torch.zeros(n + 1, dtype=self.dtype, device=self.device)

        xi[i] = torch.clone(xk[k])
        fi[i] = torch.clone(fk[k])
        si[i] = torch.clone(sk[k])

        while k != ni:

            self.history_niter += 1

            while i != n:
                xi[i + 1], fi[i + 1], si[i + 1] = minimize(sf, xi[i], fi[i], si[i], matrix[i], *args, **kwargs)
                if torch.isnan(fi[i + 1]):
                    print(f'EXIT: objective value is nan')
                    return sf, xk[:k + 1], fk[:k + 1], sk[:k + 1], mk[:k + 1]
                i += 1

            f1, s1 = fk[k], sk[k]
            f2, s2 = fi[n], si[n]
            adjust = self.to_cube(factor*xi[n] + (1.0 - factor)*(2.0*xi[n] - xk[k]))
            f3, s3 = self.objective(adjust, *args)
            self.append(adjust, f3, s3)

            if torch.isnan(f3):
                print(f'EXIT: objective value is nan')
                return sf, xk[:k + 1], fk[:k + 1], sk[:k + 1], mk[:k + 1]

            delta = -fi.diff()
            m = delta.argmax()
            delta = delta[m]

            v = xi[n] - xk[k]
            v = v/v.norm()

            if f3 >= f1 or (f1 - 2*f2 + f3)*(f1 - f2 - delta)**2 >= 0.5*delta*(f1 - f3)**2 or torch.isnan(v.sum()):
                xk[k + 1] = xi[n] if f2 < f3 else adjust
                fk[k + 1] = f2 if f2 < f3 else f3
                sk[k + 1] = s2 if f2 < f3 else s3
            else:
                xk[k + 1], fk[k + 1], sk[k + 1] = minimize(sf, xi[n], fi[n], si[n], v, *args, **kwargs)
                if torch.isnan(fk[k + 1]):
                    print(f'EXIT: objective value is nan')
                    return sf, xk[:k + 1], fk[:k + 1], sk[:k + 1], mk[:k + 1]
                matrix = torch.cat((matrix[:m], matrix[m + 1:], v.reshape(1, -1)))

            xi[0] = xk[k + 1]
            fi[0] = fk[k + 1]
            si[0] = sk[k + 1]
            mk[k + 1] = torch.clone(matrix)

            if sf != sf_min:
                sf *= dr
            if sf < sf_min:
                sf = torch.tensor(sf_min, dtype=self.dtype, device=self.device)
                if warning:
                    print(f'WARNING: minimum step fraction {sf.item()} is reached at iteration {k + 1}')

            if verbose:
                print(f'-----------------------------')
                print(f'Iteration {k + 1}:')
                print(f'-----------------------------')
                print(f'delta: {sf.cpu().numpy()}')
                print(f'knobs: {self.objective.inverse(xk[k + 1]).cpu().numpy()}')
                print(f'value: {fk[k + 1].cpu().numpy()}')
                print(f'error: {sk[k + 1].cpu().numpy()}')
                print(f'count: {self.history_neval}')
                print()

            if any((xk[k + 1] - 0.0).abs() < self.epsilon) or any((xk[k + 1] - 1.0).abs() < self.epsilon):
                if warning:
                    print(f'WARNING: value on box encounted at iteration {k + 1}')

            if termination and self.termination_knobs(xk[k + 1], xk[k], tolerance=ktol):
                print(f'EXIT: triggered knobs termination at iteration {k + 1}')
                k += 1
                break

            if termination and self.termination_value((fk[k + 1], sk[k + 1]), (fk[k], sk[k]), tolerance=otol):
                print(f'EXIT: triggered value termination at iteration {k + 1}')
                k += 1
                break

            if k + 1 == ni:
                if warning:
                    print(f'WARNING: maximum number of iterations {k + 1} is reached')

            k += 1
            i = 0
            self.history_niter = k
            sleep(pause)

        return xk[:k+1], fk[:k+1], sk[:k+1], mk[:k+1]


    def scipy(self,
              knobs:Knobs,
              *args,
              method:str='powell',
              **kwargs):
        """
        Scipy minimization wrapper for selected bounded methods.

        'powell'
        'Nelder-Mead'
        'L-BFGS-B'
        'TNC'
        'SLSQP'
        'trust-constr'

        Parameters
        ----------
        knobs: Knobs
            initial knobs
        *args:
            passed to objective
        method: str, default='powell'
            minimization method
        kwargs:
            passed to minimize

        Returns
        -------
        minimize result (OptimizeResult)

        """
        def objective(knobs, dtype:torch.dtype=self.dtype, device:torch.device=self.device):
            knobs = torch.tensor(knobs, dtype=dtype, device=device)
            value, _ = self.objective(knobs, *args)
            return value.cpu().numpy()

        if method not in ['powell', 'Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
            raise ValueError(f'MINIMIZER: unsupported method {method}')

        return minimize(
            objective,
            knobs.cpu().numpy(),
            bounds=numpy.array([numpy.zeros(self.nk), numpy.ones(self.nk)]).T,
            method=method,
            **kwargs
        )


    def torch(self,
              knobs: Knobs,
              count:int,
              optimizer,
              *args,
              **kwargs) -> tuple[Tensor, Tensor]:
        """
        Torch SGD (or other optimizer) minimization.

        Parameters
        ----------
        knobs: Knobs
            initial knobs
        count:int
            number of iterations
        optimizer:
            torch optimizer
        *args:
            passed to objective
        kwargs:
            passed to optimizer

        Returns
        -------
        knobs, value (tuple[Tensor, Tensor])

        """
        class Model(torch.nn.Module):
            def __init__(self, knobs:Knobs, minimizer:Minimizer) -> None:
                super().__init__()
                self.minimizer: Minimizer = minimizer
                self.knobs: Tensor = torch.nn.Parameter(torch.clone(knobs))
                self.history_knobs:list[Tensor] = []
                self.history_value:list[Tensor] = []
            def forward(self) -> Tensor:
                return self.minimizer.objective(self.knobs, *args)
            def train(self) -> None:
                task = optimizer(self.parameters(), **kwargs)
                for _ in range(count):
                    value, _ = self()
                    value.backward()
                    task.step()
                    task.zero_grad()
                    self.history_knobs.append(torch.clone(self.knobs.detach()))
                    self.history_value.append(torch.clone(value.detach()))
            def lbfgs(self) -> None:
                task = optimizer(self.parameters(), **kwargs)
                def closure():
                    task.zero_grad()
                    value, _ = self()
                    value.backward()
                    return value
                for _ in range(count):
                    value = task.step(closure)
                    error = torch.zeros_like(value)
                    self.history_knobs.append(torch.clone(self.knobs.detach()))
                    self.history_value.append(torch.clone(value.detach()))
        model = Model(knobs, self)
        model.lbfgs() if optimizer is torch.optim.LBFGS else model.train()
        return tuple(map(torch.stack, [model.history_knobs, model.history_value]))


    def newton(self,
               knobs:Tensor,
               jacobian:Callable,
               *args,
               count:int=1,
               factor:float=1.0,
               alpha:float=0.0) -> tuple[Tensor, Tensor]:
        """
        Newton minimization.

        Note, can be mapped over initial knobs and/or additional tensor arguments in *args

        Parameters
        ----------
        knobs: Knobs
            initial knobs
        jacobian: Callable
            jacobian function
        *args:
            passed to objective
        count: int, positive, default=1
            number of iterations
        factor: float, default=1.0
            step factor (learning rate)
        alpha: float, positive, default=0.0
            regularization alpha

        Returns
        -------
        knobs, value (tuple[Tensor, Tensor])

        """
        def function(knobs):
            value, _ = self.objective(knobs, *args)
            return value, value

        history_knobs = []
        history_value = []

        hess, grad = jacobian(jacobian(function, has_aux=True))(knobs)

        for i in range(count):

            knobs = knobs - factor * torch.pinverse(hess + alpha * torch.block_diag(*torch.ones_like(grad))) @ grad
            value, _ = function(knobs)

            history_knobs.append(knobs)
            history_value.append(value)

            if i < count - 1:
                hess, grad = jacobian(jacobian(function, has_aux=True))(knobs)

        return tuple(map(torch.stack, [history_knobs, history_value]))


    def adam(self,
             knobs:Tensor,
             jacobian:Callable,
             *args,
             count:int=1,
             lr:float=0.01,
             betas:tuple[float, float]=(0.900, 0.999),
             epsilon:float=1.0E-9) -> tuple[Tensor, Tensor]:
        """
        Adam minimization.

        Note, can be mapped over initial knobs and/or additional tensor arguments in *args

        Parameters
        ----------
        knobs: Knobs
            initial knobs
        jacobian: Callable
            jacobian function
        *args:
            passed to objective
        count: int, positive, default=1
            number of iterations
        lr: float, positive, default=0.01
            learning rate
        betas: tuple[float, float], positive, default=(0.900, 0.999)
            coefficients used for computing running averages of gradient and its square
        epsilon: float, positive, default=1.0E-9
            numerical stability epsilon

        Returns
        -------
        knobs, value (tuple[Tensor, Tensor])

        """
        def function(knobs):
            value, _ = self.objective(knobs, *args)
            return value

        b1, b2 = betas

        history_knobs = []
        history_value = []

        m1 = torch.zeros_like(knobs)
        m2 = torch.zeros_like(knobs)

        for i in range(count):

            grad = jacobian(function)(knobs)

            m1 = b1 * m1 + (1.0 - b1) * grad
            m2 = b2 * m2 + (1.0 - b2) * grad ** 2

            f1 = 1/(1 - b1 ** (i + 1))
            f2 = 1/(1 - b2 ** (i + 1))

            knobs = knobs -  lr * m1 / f1 / (torch.sqrt(m2 / f2) + epsilon)
            value = function(knobs)

            history_knobs.append(knobs)
            history_value.append(value)

        return tuple(map(torch.stack, [history_knobs, history_value]))


    @staticmethod
    def standard_errors(knobs:Knobs,
                        x:Tensor,
                        y:Tensor,
                        model:Callable[[Tensor], Tensor],
                        objective:Callable[[Tensor], Tensor],
                        estimator:Callable[[Tensor,], Tensor],
                        *args) -> Tensor:
        """
        Estimate standard errors.

        Parameters
        ----------
        knobs: Knobs
            estimated knobs
        x & y: Tensor
            input data
        model: Callable[[Tensor], Tensor]
            y = model(knobs, x, *args)
        objective: Callable[[Tensor], Tensor]
            objective(knobs, x, y, *args)
        estimator: Callable[[Tensor,], Tensor]
            variance estimator, estimator(y - model(knobs, x, *args))
        *args:
            passed to model & objective
        **kwargs:
            passed to variance estimator

        Returns
        -------
        standard errors

        """
        return torch.diag(2 * torch.linalg.pinv(torch.func.hessian(objective)(knobs, x, y, *args)) * estimator(model(knobs, x, *args) - y)).sqrt()


def main():
    pass

if __name__ == '__main__':
    main()