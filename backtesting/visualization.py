"""
Backtesting visualization utilities.
"""

import plotly.graph_objects as go
import vectorbt as vbt


def plot_equity_comparison(
    strategy_pf: vbt.Portfolio,
    buyhold_pf: vbt.Portfolio,
    title: str = "Strategy vs Buy & Hold"
) -> go.Figure:
    """
    Plot equity curves comparison using Plotly.
    
    Parameters:
    -----------
    strategy_pf : vbt.Portfolio
        Strategy portfolio
    buyhold_pf : vbt.Portfolio
        Buy & hold portfolio
    title : str
        Plot title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    strategy_equity = strategy_pf.value()
    buyhold_equity = buyhold_pf.value()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strategy_equity.index,
        y=strategy_equity.values,
        mode='lines',
        name='Strategy',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=buyhold_equity.index,
        y=buyhold_equity.values,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Portfolio Value',
        template='plotly_dark',
        hovermode='x unified',
        height=500,
        width=1000
    )
    
    return fig

