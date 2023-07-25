from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Optional
from fractions import Fraction
import numpy as np
from .pmf import ProbabilityMassFunction as PMF
from .pmf import create_cdf as C_cdf

PF = PMF
Num = int | float

@dataclass
class CumulativeDistributionFunction(object):
    cdf_data: Optional[dict] = None
    cdf_func: Optional[Callable] = None
    
    def __call__(self, nums: tuple[Num]) -> Any:
        if self.cdf_data:
            for num in nums:
                assert isinstance(num, float) or isinstance(num, int)
                if (res := self.cdf_data.get(Fraction(num), -1)) != -1:
                    return res
            raise
        if self.cdf_func:
            return self.cdf_func(nums)
        raise RuntimeError(f'没有可用的cdf {self.cdf_data} {self.cdf_func}')
CDF = CumulativeDistributionFunction

# def find_distribution(num):
#     digits = len(str(num).split('.')[-1])


def create_cdf(pf: PF) -> CDF:
    if isinstance(pf, PMF):
        assert pf.dimension == 1
        cdf_data = {distr: proba for distr, proba in zip(*C_cdf(pf))}
        return CDF(cdf_data)
    raise ValueError(f'不支持的类型: {type(pf)}')
