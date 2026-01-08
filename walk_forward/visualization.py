"""
Walk-forward visualization utilities.
"""

import plotly.graph_objects as go
import pandas as pd
import polars as pl


def plot_flow_bar(data, lookback_bars, validation_bars, testing_bars, step_size=None):
    """
    Plot walk-forward analysis windows using Plotly as horizontal bars (Gantt-style).
    Steps by testing_bars by default to ensure no gaps between test windows.
    
    Parameters:
    -----------
    data : pd.DataFrame or pl.DataFrame
        The time series data with datetime index
    lookback_bars : int
        Number of bars for training window
    validation_bars : int
        Number of bars for validation window
    testing_bars : int
        Number of bars for testing window
    step_size : int, optional
        Step size between folds. If None, uses testing_bars to ensure non-overlapping test sets.
    """
    # Convert Polars to Pandas if needed
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    
    # Default: step by testing window size to avoid gaps
    if step_size is None:
        step_size = testing_bars
    
    # Calculate how many folds we can fit
    max_start = len(data) - (validation_bars + testing_bars) + 1
    
    # Generate fold start positions
    fold_starts = list(range(lookback_bars, max_start, step_size))
    
    if len(fold_starts) == 0:
        print("Warning: No valid folds can be created with the given parameters.")
        return None
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Get data index
    data_index = data['date']
    
    # Add invisible trace to establish datetime x-axis range
    fig.add_trace(go.Scatter(
        x=[data_index.iloc[0], data_index.iloc[-1]],
        y=[-1, len(fold_starts) + 1],
        mode='markers',
        marker=dict(size=0, opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Plot each fold as a horizontal bar with three contiguous segments
    for i, val_start_idx in enumerate(fold_starts):
        fold_num = len(fold_starts) - i  # Reverse order for display
        y_center = fold_num
        y_bottom = fold_num - 0.5
        y_top = fold_num + 0.5
        
        # Training window (green)
        train_start_idx = val_start_idx - lookback_bars
        train_end_idx = val_start_idx
        
        if train_start_idx >= 0 and train_end_idx > train_start_idx:
            train_x0 = data_index.iloc[train_start_idx]
            train_x1 = data_index.iloc[train_end_idx - 1]
            
            fig.add_shape(
                type="rect",
                x0=train_x0,
                y0=y_bottom,
                x1=train_x1,
                y1=y_top,
                fillcolor='rgba(0, 128, 0, 0.7)',
                line=dict(color='black', width=0.5),
                layer='below'
            )
            
            train_x_center = train_x0 + (train_x1 - train_x0) / 2
            fig.add_trace(go.Scatter(
                x=[train_x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hovertemplate=f'<b>Fold {fold_num} - Training</b><br>' +
                             f'Start: {train_x0}<br>' +
                             f'End: {train_x1}<br>' +
                             f'Duration: {train_end_idx - train_start_idx} bars<extra></extra>',
                hoverinfo='text'
            ))
        
        # Validation window (yellow)
        val_end_idx = val_start_idx + validation_bars
        
        if val_end_idx <= len(data_index) and val_end_idx > val_start_idx:
            val_x0 = data_index.iloc[val_start_idx]
            val_x1 = data_index.iloc[val_end_idx - 1]
            
            fig.add_shape(
                type="rect",
                x0=val_x0,
                y0=y_bottom,
                x1=val_x1,
                y1=y_top,
                fillcolor='rgba(255, 255, 0, 0.7)',
                line=dict(color='black', width=0.5),
                layer='below'
            )
            
            val_x_center = val_x0 + (val_x1 - val_x0) / 2
            fig.add_trace(go.Scatter(
                x=[val_x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hovertemplate=f'<b>Fold {fold_num} - Validation</b><br>' +
                             f'Start: {val_x0}<br>' +
                             f'End: {val_x1}<br>' +
                             f'Duration: {val_end_idx - val_start_idx} bars<extra></extra>',
                hoverinfo='text'
            ))
        
        # Testing window (red)
        test_start_idx = val_end_idx
        test_end_idx = val_end_idx + testing_bars
        
        if test_end_idx <= len(data_index) and test_end_idx > test_start_idx:
            test_x0 = data_index.iloc[test_start_idx]
            test_x1 = data_index.iloc[test_end_idx - 1]
            
            fig.add_shape(
                type="rect",
                x0=test_x0,
                y0=y_bottom,
                x1=test_x1,
                y1=y_top,
                fillcolor='rgba(255, 0, 0, 0.7)',
                line=dict(color='black', width=0.5),
                layer='below'
            )
            
            test_x_center = test_x0 + (test_x1 - test_x0) / 2
            fig.add_trace(go.Scatter(
                x=[test_x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hovertemplate=f'<b>Fold {fold_num} - Testing</b><br>' +
                             f'Start: {test_x0}<br>' +
                             f'End: {test_x1}<br>' +
                             f'Duration: {test_end_idx - test_start_idx} bars<extra></extra>',
                hoverinfo='text'
            ))
    
    # Add final bar at the bottom showing all test data
    if len(fold_starts) > 0:
        first_val_start_idx = fold_starts[0]
        last_val_start_idx = fold_starts[-1]
        last_test_end_idx = last_val_start_idx + validation_bars + testing_bars
        
        if last_test_end_idx <= len(data_index) and first_val_start_idx < len(data_index):
            remaining_x0 = data_index.iloc[first_val_start_idx]
            remaining_x1 = data_index.iloc[last_test_end_idx - 1]
            
            fig.add_shape(
                type="rect",
                x0=remaining_x0,
                y0=-0.5,
                x1=remaining_x1,
                y1=0.5,
                fillcolor='rgba(139, 0, 0, 0.7)',
                line=dict(color='white', width=0.5),
                layer='below'
            )
            
            center_idx = first_val_start_idx + (last_test_end_idx - first_val_start_idx) // 2
            if center_idx >= len(data_index):
                center_idx = len(data_index) - 1
            bottom_x_center = data_index.iloc[center_idx]
            
            fig.add_trace(go.Scatter(
                x=[bottom_x_center],
                y=[0],
                mode='markers',
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hovertemplate=f'<b>Testing Period (All Folds)</b><br>' +
                             f'Start: {remaining_x0}<br>' +
                             f'End: {remaining_x1}<br>' +
                             f'Total Duration: {last_test_end_idx - first_val_start_idx} bars<extra></extra>',
                hoverinfo='text'
            ))
    
    # Update layout
    fig.update_layout(
        title='Walk-Forward Analysis Windows',
        xaxis_title='Time',
        yaxis_title='Fold Number',
        height=max(400, len(fold_starts) * 60),
        width=850,
        hovermode='closest',
        template='plotly_dark',
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[-1, len(fold_starts) + 1],
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            gridwidth=0.5
        ),
        xaxis=dict(
            type='date',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            gridwidth=0.5
        ),
        showlegend=False
    )
    
    # Add legend annotations
    fig.add_annotation(
        x=0.98, y=0.98,
        xref='paper', yref='paper',
        text='<b>Legend:</b><br>🟢 Training<br>🟡 Validation<br>🔴 Testing',
        showarrow=False,
        align='right',
        bgcolor='rgba(0, 0, 0, 0.7)',
        bordercolor='white',
        borderwidth=1
    )
    
    print(f"Number of folds created: {len(fold_starts)}")
    print(f"Fold starts: {fold_starts}")
    return fig

