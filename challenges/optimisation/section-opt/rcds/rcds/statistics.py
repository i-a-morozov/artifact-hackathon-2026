"""
Statistics module.
Compute different statistics for one-dimensional tensors.
Bootstrap and jackknife estimators.
Standardize and rescale one-dimensional tensors.

"""
from __future__ import annotations

import torch

from torch import Tensor
from typing import Optional, Callable


def mean(data:Tensor) -> Tensor:
    """
    Compute the mean for given input data.

    Parameters
    ----------
    data: Tensor
        input data

    Returns
    -------
    mean (Tensor)

    """
    return data.mean()


def variance(data:Tensor, **kwargs) -> Tensor:
    """
    Compute the variance for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    **kwargs:
        passed to torch variance function

    Returns
    -------
    variance (Tensor)

    """
    return data.var(**kwargs)


def standard_deviation(data:Tensor, **kwargs) -> Tensor:
    """
    Compute the standard deviation for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    **kwargs:
        passed variance function

    Returns
    -------
    standard deviation (Tensor)

    """
    return variance(data, **kwargs).sqrt()


def root_mean_square(data:Tensor) -> Tensor:
    """
    Compute the root mean square for given input data.

    Parameters
    ----------
    data: Tensor
        input data

    Returns
    -------
    root mean square (Tensor)

    """
    return mean(data**2).sqrt()


def weighted_mean(data:Tensor,
                  weight:Optional[Tensor]=None) -> Tensor:
    """
    Compute the weighted mean for given input data and optional weight tensor.

    Parameters
    ----------
    data: Tensor
        input data
    weight: Optional[Tensor]
        input weight tensor

    Returns
    -------
    weighted mean (Tensor)

    """
    if weight is None: 
        return mean(data)

    return 1.0/weight.sum()*(weight*data).sum()


def weighted_variance(data:Tensor,
                      weight:Optional[Tensor]=None,
                      center:Optional[Tensor]=None, 
                      **kwargs) -> Tensor:
    """
    Compute the weighted variance for given input data and weight tensor.

    Note, if center parameter is None, central tendency is estimated using weighted_mean

    Parameters
    ----------
    data: Tensor
        input data
    weight: Optional[Tensor]
        input weight tensor
    center: Optional[Tensor]
        precomputed weighted mean
    **kwargs:
        passed to variance function (used if weight tensor is None)

    Returns
    -------
    weighted variance (Tensor)

    """
    if weight is None: 
        return variance(data, **kwargs)
    
    center = weighted_mean(data, weight) if center is None else center
    
    return weight.sum()/((weight.sum())**2 - (weight**2).sum())*(weight*(data - center)**2).sum()


def trim(data:Tensor,
         *,
         f_min:float=0.1,
         f_max:float=0.1) -> Tensor:
    """
    Trim given input data.

    Note, a fraction of largest and smallest data points are dropped

    Parameters
    ----------
    data: Tensor
        input data
    f_min: float, 0.0 < f_min < 1.0, default=0.1
        min fraction to drop
    f_max: float, 0.0 < f_max < 1.0, default=0.1
        max fraction to drop

    Returns
    -------
    trimmed data (Tensor)

    """
    i_min, i_max = int(f_min*len(data)), int(len(data) - f_max*len(data))
    return data.sort().values[i_min:i_max]


def trimmed_mean(data:Tensor,
                 *,
                 f_min:float=0.1,
                 f_max:float=0.1) -> Tensor:
    """
    Compute the trimmed mean for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    f_min: float, 0.0 < f_min < 1.0, default=0.1
        min fraction to drop
    f_max: float, 0.0 < f_max < 1.0, default=0.1
        max fraction to drop

    Returns
    -------
    trimmed mean (Tensor)

    """
    i_min, i_max = int(f_min*len(data)), int(len(data) - f_max*len(data))
    return mean(trim(data, f_min=f_min, f_max=f_max))


def trimmed_variance(data:Tensor,
                     *,
                     f_min:float=0.1,
                     f_max:float=0.1, 
                     **kwargs) -> Tensor:
    """
    Compute the trimmed variance for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    f_min: float, 0.0 < f_min < 1.0, default=0.1
        min fraction to drop
    f_max: float, 0.0 < f_max < 1.0, default=0.1
        max fraction to drop
    **kwargs:
        passed to variance function

    Returns
    -------
    trimmed variance (Tensor)

    """
    return variance(trim(data, f_min=f_min, f_max=f_max), **kwargs)


def winsorize(data:Tensor,
              *,
              f_min:float=0.1,
              f_max:float=0.1) -> Tensor:
    """
    Winsorize given input data.

    Note, a fraction of largest and smallest data points are replaced by edge values

    Parameters
    ----------
    data: Tensor
        input data
    f_min: float, 0.0 < f_min < 1.0, default=0.1
        min fraction to drop
    f_max: float, 0.0 < f_max < 1.0, default=0.1
        max fraction to drop

    Returns
    -------
    winsorized data (Tensor)

    """
    i_min, i_max = int(f_min*len(data)), int(len(data) - f_max*len(data))
    data = data.sort().values
    min_value, max_value = data[i_min], data[i_max]
    data[:i_min] = min_value
    data[i_max:] = max_value
    return data


def winsorized_mean(data:Tensor,
                    *,
                    f_min:float=0.1,
                    f_max:float=0.1) -> Tensor:
    """
    Compute the winsorized mean for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    f_min: float, 0.0 < f_min < 1.0, default=0.1
        min fraction to drop
    f_max: float, 0.0 < f_max < 1.0, default=0.1
        max fraction to drop

    Returns
    -------
    winsorized mean (Tensor)

    """
    return mean(winsorize(data, f_min=f_min, f_max=f_max))


def winsorized_variance(data:Tensor,
                        *,
                        f_min:float=0.1,
                        f_max:float=0.1,
                        **kwargs) -> Tensor:
    """
    Compute the winsorized variance for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    f_min: float, 0.0 < f_min < 1.0, default=0.1
        min fraction to drop
    f_max: float, 0.0 < f_max < 1.0, default=0.1
        max fraction to drop
    **kwargs:
        passed to variance function

    Returns
    -------
    winsorized variance (Tensor)

    """
    return variance(winsorize(data, f_min=f_min, f_max=f_max), **kwargs)


def median(data:Tensor,
           **kwargs) -> Tensor:
    """
    Compute the median for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    **kwargs:
        passed to torch quantile

    Returns
    -------
    median (Tensor)

    """
    return data.quantile(0.5, **kwargs)


def median_deviation(data:Tensor,
                     *,
                     center:Optional[Tensor]=None,
                     **kwargs) -> Tensor:
    """
    Compute the median absolute deviation from the median for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    center: Optional[Tensor]
        precomputed median
    **kwargs:
        passed to median function

    Returns
    -------
    median deviation (Tensor)

    """
    center = median(data, **kwargs) if center is None else center

    return median((data - center).abs())


def biweight_midvariance(data:Tensor,
                         *,
                         scale:float=10.0,
                         center:Optional[Tensor]=None,
                         deviation:Optional[Tensor]=None,
                         **kwargs) -> Tensor:
    """
    Compute the biweight midvariance for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    scale: float, positive, default=10.0
        scaling parameter (number of median deviations)
    center: Optional[Tensor]
        precomputed median
    deviation: Optional[Tensor]
        precomputed median absolute deviation
    **kwargs:
        passed to median and median_deviation functions

    Returns
    -------
    biweight midvariance (Tensor)

    """
    center = median(data, **kwargs) if center is None else center

    deviation = median_deviation(data, center=center, **kwargs) if deviation is None else deviation

    weight = (1.0 - ((data - center)/(scale*deviation))**2)
    weight *= weight > 0.0

    return (len(data)*(weight**4*(data - center)**2).sum()/(weight*(5.0*weight - 4.0)).sum()**2)


def quantile(data:Tensor,
             fraction:Tensor,
             **kwargs) -> Tensor:
    """
    Compute the quantile for given input data and list of fractions.

    Parameters
    ----------
    data: Tensor
        input data
    fraction: Tensor
        list of fractions
    **kwargs:
        passed to torch quantile

    Returns
    -------
    quantile (Tensor)

    """
    return torch.quantile(data, fraction, **kwargs)


def quartiles(data:Tensor,
              **kwargs) -> Tensor:
    """
    Compute (0.25, 0.50, 0.75) quartiles for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    **kwargs:
        passed to quantile

    Returns
    -------
    quartiles (Tensor)

    """
    return quantile(data, torch.tensor([0.25, 0.50, 0.75], dtype=data.dtype, device=data.device), **kwargs)


def interquartile_range(data:Tensor,
                        **kwargs) -> Tensor:
    """
    Compute interquartile range for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    **kwargs:
        passed to quartiles function

    Returns
    -------
    interquartile range (Tensor)

    """
    q_min, _, q_max = quartiles(data, **kwargs)

    return (q_max - q_min)


def whiskers(data:Tensor,
             *,
             factor:float=1.5,
             **kwargs) -> Tensor:
    """
    Compute 'whiskers' for given input data.

    Parameters
    ----------
    data: Tensor
        input data
    factor: float, positive, default=1.5
        scaling factor
    **kwargs:
        passed to quartiles function

    Returns
    -------
    whiskers (Tensor)

    """
    q_min, _, q_max = quartiles(data, **kwargs)

    iqr = (q_max - q_min)

    return torch.stack([q_min - factor*iqr, q_max + factor*iqr])


def bootstrap(estimator:Callable[[Tensor], Tensor],
              *,
              limit:Optional[int]=None,
              count:Optional[int]=None,
              **kwargs) -> Callable[[Tensor], Tensor]:
    """
    Bootstrap given estimator.

    Parameters
    ----------
    estimator: Callable[[Tensor], Tensor]
        input estimator to bootstrap
    limit: Optional[int]
        sample size to use
    count: Optional[int]
        number of samples to use
    **kwargs:
        passed to estimator

    Returns
    -------
    bootstapped estimator (Callable[[Tensor], Tensor])

    """
    def closure(data:Tensor, *, limit:Optional[int]=limit, count:Optional[int]=count, **kwargs) -> Tensor:
        limit = len(data) if limit is None else limit
        count = len(data) if count is None else count
        index = torch.randint(limit, (count, limit), dtype=torch.int64, device=data.device)
        return torch.func.vmap(lambda index: estimator(data[index], **kwargs))(index)

    return closure


def jackknife(estimator:Callable[[Tensor], Tensor],
              **kwargs,
             ) -> Callable[[Tensor], Tensor]:
    """
    Jackknife given estimator.

    Parameters
    ----------
    estimator: Callable[[Tensor], Tensor]
        input estimator to bootstrap
    **kwargs:
        passed to estimator

    Returns
    -------
    Jackknifed estimator (Callable[[Tensor], Tensor])

    """
    def closure(data:Tensor, **kwargs) -> Tensor:
        index = torch.arange(len(data), dtype=torch.int64, device=data.device)
        index = torch.stack([index[index != i] for i in index])
        return torch.func.vmap(lambda index: estimator(data[index], **kwargs))(index)

    return closure


def standardize(data:Tensor,
                *,
                center_estimator:Callable[[Tensor], Tensor]=mean,
                spread_estimator:Callable[[Tensor], Tensor]=variance) -> Tensor:
    """
    Standardize input data.

    Note, default center/spread are mean/variance

    Parameters
    ----------
    data: Tensor
        input data
    center_estimator: Callable[[Tensor], Tensor], default=mean
        center estimator
    spread_estimator: Callable[[Tensor], Tensor], default=varianve
        spread estimator

    Returns
    -------
    standardized data (Tensor)

    """
    return (data - center_estimator(data))/spread_estimator(data).sqrt()


def rescale(data:Tensor,
            *,
            range_min:Optional[float]=None,
            range_max:Optional[float]=None,
            scale_min:Optional[float]=None,
            scale_max:Optional[float]=None) -> Tensor:
    """
    Rescale input data.

    Note, (scale_min, scale_max) = (-1, +1) and (range_min, range_max) = (min(data), max(data))
    Default min/max of input data is mapped to -1/+1

    Parameters
    ----------
    data: Tensor
        input data
    range_min: Optional[float]
        input min range
    range_max: Optional[float]
        input max range
    scale_min: Optional[float]
        output min range
    scale_max: Optional[float]
        output max range

    Returns
    -------
    scaled data (Tensor)

    """
    range_min = data.min() if range_min is None else range_min
    range_max = data.max() if range_max is None else range_max

    scale_min = -1.0 if scale_min is None else scale_min
    scale_max = +1.0 if scale_max is None else scale_max

    return (data*(scale_max - scale_min)/(range_max - range_min) + (range_max*scale_min - range_min*scale_max)/(range_max - range_min))


def main():
    pass

if __name__ == '__main__':
    main()