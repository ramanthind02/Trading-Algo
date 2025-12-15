"""
Portfolio Analytics and Visualization

This module provides a wrapper around QuantStats with custom extensions
for walk-forward analysis and trading strategy evaluation.

Since QuantStats is not actively maintained, this wrapper allows us to:
1. Use QuantStats functionality where it works well
2. Customize and extend it for our specific needs
3. Maintain our own local implementations if needed

Author: Trading Research Team
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import warnings

# Import QuantStats (installed in venv)
try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False
    warnings.warn("QuantStats not installed. Some functionality will be limited.")


class PortfolioAnalytics:
    """
    Wrapper class for portfolio analytics and visualization.
    
    Provides a clean interface to QuantStats functionality with custom extensions
    for walk-forward analysis and trading strategy evaluation.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns (daily, indexed by date)
    benchmark : Optional[pd.Series], default=None
        Benchmark returns for comparison
    """
    
    def __init__(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None
    ):
        self.returns = returns
        self.benchmark = benchmark
        
        if HAS_QUANTSTATS:
            # Extend pandas with QuantStats methods
            qs.extend_pandas()
    
    def compute_metrics(self, rf: float = 0.0) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Parameters
        ----------
        rf : float, default=0.0
            Risk-free rate (annualized)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics
        """
        if not HAS_QUANTSTATS:
            return self._compute_basic_metrics(rf)
        
        metrics = {
            # Returns
            'total_return': qs.stats.comp(self.returns),
            'cagr': qs.stats.cagr(self.returns),
            'avg_return': qs.stats.avg_return(self.returns),
            
            # Risk
            'volatility': qs.stats.volatility(self.returns),
            'max_drawdown': qs.stats.max_drawdown(self.returns),
            'var': qs.stats.var(self.returns),
            'cvar': qs.stats.cvar(self.returns),
            
            # Risk-adjusted
            'sharpe': qs.stats.sharpe(self.returns, rf=rf),
            'sortino': qs.stats.sortino(self.returns, rf=rf),
            'calmar': qs.stats.calmar(self.returns),
            
            # Win/Loss
            'win_rate': qs.stats.win_rate(self.returns),
            'profit_factor': qs.stats.profit_factor(self.returns),
            'payoff_ratio': qs.stats.payoff_ratio(self.returns),
            
            # Other
            'skew': qs.stats.skew(self.returns),
            'kurtosis': qs.stats.kurtosis(self.returns),
        }
        
        return metrics
    
    def _compute_basic_metrics(self, rf: float = 0.0) -> Dict[str, float]:
        """
        Fallback: Compute basic metrics without QuantStats.
        
        Parameters
        ----------
        rf : float, default=0.0
            Risk-free rate (annualized)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of basic performance metrics
        """
        returns = self.returns.dropna()
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized metrics
        n_periods = len(returns)
        years = n_periods / 252  # Assuming daily returns
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns - rf / 252
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
        sortino = (returns.mean() * 252 - rf) / downside_std
        
        # Max drawdown
        from metrics.risk import max_drawdown as compute_max_drawdown
        max_drawdown = compute_max_drawdown(returns=returns)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
        }
    
    def plot_snapshot(
        self,
        title: str = "Strategy Performance",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance snapshot.
        
        Parameters
        ----------
        title : str, default="Strategy Performance"
            Plot title
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        save_path : Optional[str], default=None
            If provided, save figure to this path
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        if HAS_QUANTSTATS:
            # Use QuantStats snapshot
            fig = qs.plots.snapshot(
                self.returns,
                title=title,
                show=False
            )
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
        else:
            # Fallback: Create basic snapshot
            return self._plot_basic_snapshot(title, figsize, save_path)
    
    def _plot_basic_snapshot(
        self,
        title: str,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> plt.Figure:
        """
        Fallback: Create basic performance snapshot without QuantStats.
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Plot 1: Cumulative returns
        from metrics.equity import cumulative_returns
        from metrics.risk import drawdown_series
        
        ax = axes[0]
        cumulative = cumulative_returns(self.returns)
        ax.plot(cumulative.index, cumulative.values, linewidth=2, color='#2ecc71')
        ax.fill_between(cumulative.index, 1, cumulative.values, alpha=0.3, color='#2ecc71')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Cumulative Return', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax = axes[1]
        drawdown = drawdown_series(returns=self.returns)
        ax.fill_between(drawdown.index, 0, drawdown.values, alpha=0.7, color='#e74c3c')
        ax.set_ylabel('Drawdown', fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_html_report(
        self,
        output_path: str,
        title: str = "Strategy Report"
    ) -> None:
        """
        Generate comprehensive HTML tearsheet.
        
        Parameters
        ----------
        output_path : str
            Path to save HTML report
        title : str, default="Strategy Report"
            Report title
        """
        if not HAS_QUANTSTATS:
            raise NotImplementedError(
                "HTML reports require QuantStats. Install with: pip install quantstats"
            )
        
        qs.reports.html(
            self.returns,
            benchmark=self.benchmark,
            output=output_path,
            title=title
        )
    
    @staticmethod
    def from_walkforward_results(
        results_df: pd.DataFrame,
        strategy: str = 'long'
    ) -> 'PortfolioAnalytics':
        """
        Create PortfolioAnalytics from walk-forward analysis results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Walk-forward results with columns: 'test_start', 'mean_return', 'n_trades'
        strategy : str, default='long'
            Strategy name for labeling
            
        Returns
        -------
        PortfolioAnalytics
            Analytics object for the strategy
        """
        # Convert walk-forward results to a return series
        # Use test_start as the index and mean_return as the value
        returns = pd.Series(
            results_df['mean_return'].values,
            index=pd.to_datetime(results_df['test_start'])
        )
        
        return PortfolioAnalytics(returns=returns)


# Convenience functions for quick analysis
def quick_metrics(returns: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """
    Quick computation of performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    rf : float, default=0.0
        Risk-free rate (annualized)
        
    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics
    """
    analytics = PortfolioAnalytics(returns)
    return analytics.compute_metrics(rf=rf)


def quick_snapshot(
    returns: pd.Series,
    title: str = "Strategy Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Quick performance snapshot plot.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    title : str, default="Strategy Performance"
        Plot title
    save_path : Optional[str], default=None
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    analytics = PortfolioAnalytics(returns)
    return analytics.plot_snapshot(title=title, save_path=save_path)
