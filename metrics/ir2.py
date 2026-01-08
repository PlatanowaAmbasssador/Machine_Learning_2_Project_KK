"""
IR2 (Information Ratio Squared) calculation for trading evaluation.
"""

import numpy as np


def equity_from_returns(returns, start=1.0):
    """Convert a return series into an equity curve."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.array([start], dtype=float)
    eq = np.empty(returns.size + 1, dtype=float)
    eq[0] = start
    eq[1:] = start * np.cumprod(1.0 + returns)
    return eq


def EquityCurve_na_StopyZwrotu(tab):
    """Convert equity curve to returns."""
    return [(tab[i+1]/tab[i])-1 for i in range(len(tab)-1)]


def ARC(tab):
    """Annualized Return Compound."""
    r = EquityCurve_na_StopyZwrotu(tab)
    if not r:
        return 0
    a = 1.0
    for x in r[:-1]:
        a *= (1+x)
    a = 0 if a <= 0 else (a**(252/len(tab)) - 1)
    return 100*a


def MaximumDrawdown(tab):
    """Calculate maximum drawdown percentage."""
    r = np.array(EquityCurve_na_StopyZwrotu(tab))
    if r.size == 0:
        return 0
    cum = np.cumprod(1+r)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum)/peak
    return float(dd.max()*100)


def ASD(tab):
    """Annualized Standard Deviation."""
    r = EquityCurve_na_StopyZwrotu(tab)
    return float((252**0.5)*np.std(r)*100) if len(r) > 1 else 0.0


def sgn(x):
    """Sign function."""
    return 0 if x == 0 else int(abs(x)/x)


def IR2(tab):
    """
    Information Ratio Squared.
    
    IR2 = (ARC^2 * sign(ARC)) / (ASD * MDD)
    
    Parameters:
    -----------
    tab : np.ndarray or list
        Equity curve values
        
    Returns:
    --------
    ir2 : float
        IR2 metric value
    """
    aSD, ret, md = ASD(tab), ARC(tab), MaximumDrawdown(tab)
    denom = aSD * md
    val = ((ret**2)*sgn(ret)/denom) if denom else 0
    return max(val, 0)


def calculate_ir2_from_returns(returns):
    """
    Calculate IR2 from a return series.
    
    Parameters:
    -----------
    returns : np.ndarray
        Strategy returns
        
    Returns:
    --------
    ir2 : float
        IR2 metric value
    """
    eq = equity_from_returns(returns, start=1.0)
    return IR2(eq)

