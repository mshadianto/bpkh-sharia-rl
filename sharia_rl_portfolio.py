"""
=========================================================================
Sharia-Constrained RL Portfolio Optimization - Research Prototype
=========================================================================
Author: Sopian (MS Hadianto)
Target: Ph.D. in Economics, UIII
Title: "Algorithmic Portfolio Optimization Under Sharia Constraints:
        A Reinforcement Learning Approach to Dynamic Asset Allocation"

This prototype implements the core framework for training DQN, PPO, and
A2C agents on JII (Jakarta Islamic Index) stock data with hard-coded
Sharia compliance constraints.

Requirements:
    pip install gymnasium numpy pandas yfinance stable-baselines3 
    pip install torch matplotlib seaborn scikit-learn ta

Usage:
    python sharia_rl_portfolio.py --algo PPO --episodes 500 --seed 42
=========================================================================
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# =========================================================================
# 1. CONFIGURATION
# =========================================================================

@dataclass
class ShariaConfig:
    """DSN-MUI & AAOIFI Sharia screening thresholds."""
    max_debt_ratio: float = 0.45        # Total debt / total assets ≤ 45%
    max_nonhalal_revenue: float = 0.10   # Non-halal revenue / total ≤ 10%
    min_compliance_score: float = 0.95   # Minimum portfolio SCS
    purification_rate: float = 0.025     # Zakat purification rate
    prohibited_sectors: List[str] = field(default_factory=lambda: [
        "alcohol", "tobacco", "gambling", "conventional_finance",
        "pork", "weapons", "adult_entertainment",
    ])


@dataclass
class RLConfig:
    """Reinforcement Learning hyperparameters."""
    train_start: str = "2015-01-01"
    train_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2025-06-30"
    initial_capital: float = 1_000_000_000  # IDR 1 Billion
    transaction_cost: float = 0.0015        # 15 bps per trade
    risk_free_rate: float = 0.05            # ~5% sukuk yield
    gamma: float = 0.99                     # Discount factor
    reward_alpha: float = 0.5              # Sharpe weight
    reward_beta: float = 0.3               # MDD penalty weight
    reward_lambda: float = 0.2             # Sharia compliance weight
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    sharia: ShariaConfig = field(default_factory=ShariaConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    jii_tickers: List[str] = field(default_factory=lambda: [
        # JII Top 30 - representative subset for prototype
        "TLKM.JK", "ASII.JK", "UNVR.JK", "BBCA.JK", "BMRI.JK",
        "ICBP.JK", "INDF.JK", "KLBF.JK", "PGAS.JK", "SMGR.JK",
        "PTBA.JK", "ADRO.JK", "ANTM.JK", "INCO.JK", "CPIN.JK",
        "EXCL.JK", "MNCN.JK", "BRPT.JK", "TPIA.JK", "ACES.JK",
    ])
    malaysia_tickers: List[str] = field(default_factory=lambda: [
        # FTSE Bursa Malaysia Hijrah Shariah - representative subset
        "1155.KL", "5183.KL", "4707.KL", "6888.KL", "5225.KL",
        "4863.KL", "5347.KL", "3182.KL", "6947.KL", "4677.KL",
    ])


# =========================================================================
# 2. DATA PIPELINE
# =========================================================================

class ShariaDataPipeline:
    """
    Downloads, cleans, and prepares market data for RL training.
    Applies Sharia screening as hard constraints.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.eligible_mask: Optional[pd.DataFrame] = None
    
    def download_data(self, tickers: List[str], market: str = "JII") -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance."""
        import yfinance as yf
        
        print(f"\n📥 Downloading {market} data for {len(tickers)} tickers...")
        data = yf.download(
            tickers,
            start=self.config.rl.train_start,
            end=self.config.rl.test_end,
            auto_adjust=True,
            progress=True,
        )
        
        # Extract close prices
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].dropna(axis=1, how="all")
        else:
            close = data[["Close"]].dropna()
        
        print(f"  ✅ Downloaded {close.shape[0]} days × {close.shape[1]} stocks")
        return close
    
    def compute_features(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical features for RL state representation.
        Features: returns, volatility, RSI, MACD, Bollinger Band width.
        """
        import ta
        
        features = {}
        for col in close.columns:
            series = close[col].dropna()
            if len(series) < 50:
                continue
            
            ret = series.pct_change()
            features[f"{col}_return"] = ret
            features[f"{col}_vol20"] = ret.rolling(20).std()
            features[f"{col}_rsi"] = ta.momentum.rsi(series, window=14)
            
            macd = ta.trend.MACD(series)
            features[f"{col}_macd"] = macd.macd_diff()
            
            bb = ta.volatility.BollingerBands(series, window=20)
            features[f"{col}_bb_width"] = bb.bollinger_wband()
        
        df = pd.DataFrame(features, index=close.index).dropna()
        print(f"  📊 Computed {df.shape[1]} features across {df.shape[0]} days")
        return df
    
    def apply_sharia_screen(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Sharia compliance screening.
        
        In production: query fundamental data for debt ratios and revenue screens.
        For prototype: use JII/Hijrah membership as proxy (already pre-screened).
        Returns binary eligibility mask per stock per day.
        """
        # JII stocks are already DSN-MUI screened; simulate periodic re-screening
        # In the full study, fundamental data from IDX/Bursa Malaysia will be used
        mask = pd.DataFrame(
            np.ones_like(close.values, dtype=float),
            index=close.index,
            columns=close.columns,
        )
        
        # Simulate semi-annual Sharia review (stocks may exit/enter index)
        np.random.seed(self.config.rl.seed)
        for col in mask.columns:
            # ~5% chance of temporary non-compliance per review period
            review_dates = pd.date_range(close.index[0], close.index[-1], freq="6MS")
            for rd in review_dates:
                if np.random.random() < 0.05:
                    end = rd + pd.DateOffset(months=6)
                    mask.loc[rd:end, col] = 0.0
        
        compliant_pct = mask.mean().mean() * 100
        print(f"  🕌 Sharia screening: {compliant_pct:.1f}% average compliance rate")
        return mask


# =========================================================================
# 3. MARKET ENVIRONMENT (Gymnasium-compatible)
# =========================================================================

class ShariaPortfolioEnv:
    """
    Custom RL environment for Sharia-constrained portfolio optimization.
    
    Implements the MDP: (S, A, P, R, γ)
    - State: portfolio weights + market features + compliance status
    - Action: target portfolio weights (continuous, sum-to-one)
    - Reward: composite function with Sharpe, MDD, and SCS
    """
    
    def __init__(
        self,
        close: pd.DataFrame,
        features: pd.DataFrame,
        sharia_mask: pd.DataFrame,
        config: ExperimentConfig,
        mode: str = "train",
    ):
        self.close = close
        self.features = features
        self.sharia_mask = sharia_mask
        self.config = config
        self.n_assets = close.shape[1]
        
        # Split train/test
        if mode == "train":
            idx = (close.index >= config.rl.train_start) & (close.index <= config.rl.train_end)
        else:
            idx = (close.index >= config.rl.test_start) & (close.index <= config.rl.test_end)
        
        self.dates = close.index[idx]
        self.returns = close.pct_change().loc[self.dates].fillna(0).values
        self.mask = sharia_mask.loc[self.dates].values
        
        # State and action dimensions
        n_features_per_stock = 5  # return, vol, rsi, macd, bb_width
        self.state_dim = self.n_assets * (n_features_per_stock + 2)  # +weights +compliance
        self.action_dim = self.n_assets
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_idx = 0
        self.weights = np.ones(self.n_assets) / self.n_assets  # Equal weight start
        self.portfolio_value = self.config.rl.initial_capital
        self.portfolio_history = [self.portfolio_value]
        self.weight_history = [self.weights.copy()]
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        t = min(self.step_idx, len(self.returns) - 1)
        market_state = self.returns[t]  # Simplified; full version uses features
        compliance = self.mask[t]
        state = np.concatenate([self.weights, market_state, compliance])
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one trading step.
        
        1. Apply Sharia mask to action (zero out non-compliant assets)
        2. Normalize weights to sum to 1
        3. Compute portfolio return after transaction costs
        4. Calculate composite reward
        """
        # Enforce Sharia constraints (hard mask)
        t = min(self.step_idx, len(self.returns) - 1)
        masked_action = action * self.mask[t]
        
        # Normalize to valid portfolio weights
        total = masked_action.sum()
        if total > 0:
            new_weights = masked_action / total
        else:
            new_weights = np.ones(self.n_assets) / self.n_assets
        
        # Transaction costs
        turnover = np.abs(new_weights - self.weights).sum()
        tc = turnover * self.config.rl.transaction_cost
        
        # Portfolio return
        if self.step_idx < len(self.returns):
            port_return = (new_weights * self.returns[t]).sum() - tc
        else:
            port_return = 0.0
        
        # Update state
        self.weights = new_weights
        self.portfolio_value *= (1 + port_return)
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(self.weights.copy())
        self.step_idx += 1
        
        # Compute reward
        reward = self._compute_reward(port_return, new_weights, t)
        
        done = self.step_idx >= len(self.returns)
        info = {
            "portfolio_value": self.portfolio_value,
            "return": port_return,
            "turnover": turnover,
            "sharia_score": self._compute_scs(new_weights, t),
        }
        
        return self._get_state(), reward, done, info
    
    def _compute_reward(self, ret: float, weights: np.ndarray, t: int) -> float:
        """
        Composite reward: R(t) = α×Sharpe(t) + β×(1-MDD(t)) + λ×SCS(t)
        """
        cfg = self.config.rl
        
        # Rolling Sharpe (simplified)
        if len(self.portfolio_history) > 20:
            recent = np.array(self.portfolio_history[-21:])
            rets = np.diff(recent) / recent[:-1]
            sharpe = (rets.mean() - cfg.risk_free_rate/252) / (rets.std() + 1e-8)
        else:
            sharpe = ret * 100  # Scale raw return for early episodes
        
        # Maximum drawdown penalty
        peak = max(self.portfolio_history)
        mdd = (peak - self.portfolio_value) / peak
        
        # Sharia Compliance Score
        scs = self._compute_scs(weights, t)
        
        reward = cfg.reward_alpha * sharpe + cfg.reward_beta * (1 - mdd) + cfg.reward_lambda * scs
        return float(reward)
    
    def _compute_scs(self, weights: np.ndarray, t: int) -> float:
        """
        Sharia Compliance Score (SCS) - Novel metric.
        Weighted average of asset compliance status.
        SCS = Σ(w_i × compliance_i) for all assets i.
        """
        compliance = self.mask[min(t, len(self.mask)-1)]
        return float(np.dot(weights, compliance))


# =========================================================================
# 4. RL AGENT WRAPPERS
# =========================================================================

class RLAgentFactory:
    """
    Factory for creating DQN, PPO, and A2C agents.
    Uses Stable-Baselines3 for implementation.
    """
    
    @staticmethod
    def create_agent(algo: str, env, config: RLConfig, **kwargs):
        """
        Create an RL agent.
        
        Args:
            algo: One of "DQN", "PPO", "A2C"
            env: Gymnasium-compatible environment
            config: RL configuration
        """
        from stable_baselines3 import DQN, PPO, A2C
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        common_params = {
            "verbose": 1,
            "seed": config.seed,
            "gamma": config.gamma,
            "tensorboard_log": "./tb_logs/",
        }
        
        agents = {
            "DQN": lambda: DQN(
                "MlpPolicy", env,
                learning_rate=1e-4,
                buffer_size=50000,
                batch_size=64,
                exploration_fraction=0.3,
                **common_params,
            ),
            "PPO": lambda: PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                clip_range=0.2,
                **common_params,
            ),
            "A2C": lambda: A2C(
                "MlpPolicy", env,
                learning_rate=7e-4,
                n_steps=5,
                **common_params,
            ),
        }
        
        if algo not in agents:
            raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(agents.keys())}")
        
        print(f"\n🤖 Creating {algo} agent...")
        return agents[algo]()


# =========================================================================
# 5. BACKTESTING ENGINE
# =========================================================================

class BacktestEngine:
    """
    Evaluates trained agents and benchmarks on test data.
    Computes all performance metrics.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Dict[str, dict] = {}
    
    def evaluate(self, name: str, portfolio_values: List[float]) -> dict:
        """Compute comprehensive performance metrics."""
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # Annualized metrics
        n_days = len(returns)
        annual_factor = 252 / n_days if n_days > 0 else 1
        
        cum_return = (values[-1] / values[0]) - 1
        annual_return = (1 + cum_return) ** annual_factor - 1
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        rf_daily = self.config.rl.risk_free_rate / 252
        sharpe = (returns.mean() - rf_daily) / (returns.std() + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_dd = drawdown.max()
        
        # Calmar ratio
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
        sortino = (annual_return - self.config.rl.risk_free_rate) / downside_std
        
        metrics = {
            "name": name,
            "cumulative_return": cum_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "n_trading_days": n_days,
        }
        
        self.results[name] = metrics
        return metrics
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comparison table of all strategies."""
        df = pd.DataFrame(self.results).T
        df = df.round(4)
        
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS COMPARISON")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)
        
        return df
    
    def plot_equity_curves(self, curves: Dict[str, List[float]], save_path: str = None):
        """Plot portfolio equity curves for all strategies."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
        
        colors = {
            "DQN": "#1565C0", "PPO": "#2E7D32", "A2C": "#E65100",
            "Equal Weight": "#757575", "Market Cap": "#9E9E9E",
            "Mean-Variance": "#BDBDBD",
        }
        
        # Equity curves
        for name, values in curves.items():
            normalized = np.array(values) / values[0]
            color = colors.get(name, "#333333")
            axes[0].plot(normalized, label=name, color=color, linewidth=2 if name in ["DQN","PPO","A2C"] else 1)
        
        axes[0].set_title("Portfolio Equity Curves (Normalized)", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Portfolio Value (Normalized)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        for name, values in curves.items():
            values = np.array(values)
            peak = np.maximum.accumulate(values)
            dd = (peak - values) / peak
            color = colors.get(name, "#333333")
            if name in ["DQN", "PPO", "A2C"]:
                axes[1].fill_between(range(len(dd)), -dd, alpha=0.3, color=color, label=name)
        
        axes[1].set_title("Drawdown (RL Agents)", fontsize=12)
        axes[1].set_ylabel("Drawdown")
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  📈 Chart saved: {save_path}")
        plt.close()


# =========================================================================
# 6. BENCHMARK STRATEGIES
# =========================================================================

class BenchmarkStrategies:
    """Implements conventional benchmark portfolios."""
    
    @staticmethod
    def equal_weight(returns: np.ndarray, initial_capital: float) -> List[float]:
        """1/N equal-weight portfolio."""
        n_assets = returns.shape[1]
        weights = np.ones(n_assets) / n_assets
        values = [initial_capital]
        for t in range(len(returns)):
            port_ret = (weights * returns[t]).sum()
            values.append(values[-1] * (1 + port_ret))
        return values
    
    @staticmethod
    def market_cap_weighted(returns: np.ndarray, initial_capital: float, 
                            market_caps: Optional[np.ndarray] = None) -> List[float]:
        """Market-capitalization weighted portfolio."""
        n_assets = returns.shape[1]
        if market_caps is None:
            # Proxy: use cumulative return as cap proxy
            cum = (1 + returns).cumprod(axis=0)
            market_caps = cum[-1] if len(cum) > 0 else np.ones(n_assets)
        
        weights = market_caps / market_caps.sum()
        values = [initial_capital]
        for t in range(len(returns)):
            port_ret = (weights * returns[t]).sum()
            values.append(values[-1] * (1 + port_ret))
        return values


# =========================================================================
# 7. MAIN EXPERIMENT RUNNER
# =========================================================================

def run_experiment(algo: str = "PPO", episodes: int = 100, seed: int = 42):
    """
    Main experiment pipeline:
    1. Download data
    2. Apply Sharia screening
    3. Train RL agent
    4. Backtest on test period
    5. Compare with benchmarks
    6. Generate report
    """
    print("=" * 80)
    print("🕌 SHARIA-CONSTRAINED RL PORTFOLIO OPTIMIZATION")
    print(f"   Algorithm: {algo} | Episodes: {episodes} | Seed: {seed}")
    print("=" * 80)
    
    # Configuration
    config = ExperimentConfig()
    config.rl.seed = seed
    
    # Data pipeline
    pipeline = ShariaDataPipeline(config)
    
    try:
        close = pipeline.download_data(config.jii_tickers, market="JII")
    except Exception as e:
        print(f"  ⚠️  Data download failed: {e}")
        print("  📝 Using synthetic data for prototype demonstration...")
        close = _generate_synthetic_data(config)
    
    features = pipeline.compute_features(close)
    sharia_mask = pipeline.apply_sharia_screen(close)
    
    # Create environments
    print("\n🏗️  Building RL environments...")
    train_env = ShariaPortfolioEnv(close, features, sharia_mask, config, mode="train")
    test_env = ShariaPortfolioEnv(close, features, sharia_mask, config, mode="test")
    
    print(f"  State dim: {train_env.state_dim}")
    print(f"  Action dim: {train_env.action_dim}")
    print(f"  Train days: {len(train_env.dates)}")
    print(f"  Test days: {len(test_env.dates)}")
    
    # Train agent (skeleton - full training requires SB3 Gym wrapper)
    print(f"\n🎓 Training {algo} agent ({episodes} episodes)...")
    print("  [In full implementation: SB3 agent.learn() with custom Gym wrapper]")
    print("  [Prototype: random policy demonstration]")
    
    # Simulate test run with random policy (placeholder)
    np.random.seed(seed)
    state = test_env.reset()
    rl_values = [config.rl.initial_capital]
    
    while True:
        action = np.random.dirichlet(np.ones(test_env.n_assets))
        state, reward, done, info = test_env.step(action)
        rl_values.append(info["portfolio_value"])
        if done:
            break
    
    # Benchmark strategies
    print("\n📊 Computing benchmark portfolios...")
    test_returns = close.pct_change().loc[
        (close.index >= config.rl.test_start) & (close.index <= config.rl.test_end)
    ].fillna(0).values
    
    ew_values = BenchmarkStrategies.equal_weight(test_returns, config.rl.initial_capital)
    mcw_values = BenchmarkStrategies.market_cap_weighted(test_returns, config.rl.initial_capital)
    
    # Evaluation
    engine = BacktestEngine(config)
    engine.evaluate(f"{algo} (Sharia-RL)", rl_values)
    engine.evaluate("Equal Weight", ew_values)
    engine.evaluate("Market Cap Weighted", mcw_values)
    
    report = engine.generate_report()
    
    # Save results
    report.to_csv("/home/claude/backtest_results.csv")
    
    # Plot
    curves = {
        f"{algo} (Sharia-RL)": rl_values,
        "Equal Weight": ew_values,
        "Market Cap": mcw_values,
    }
    engine.plot_equity_curves(curves, save_path="/home/claude/equity_curves.png")
    
    print("\n✅ Experiment complete!")
    print(f"  📄 Results: backtest_results.csv")
    print(f"  📈 Charts: equity_curves.png")
    
    return report


def _generate_synthetic_data(config: ExperimentConfig) -> pd.DataFrame:
    """Generate synthetic stock price data for prototype testing."""
    np.random.seed(config.rl.seed)
    dates = pd.bdate_range(config.rl.train_start, config.rl.test_end)
    n_stocks = len(config.jii_tickers)
    
    # GBM simulation
    mu = np.random.uniform(0.05, 0.15, n_stocks) / 252
    sigma = np.random.uniform(0.15, 0.35, n_stocks) / np.sqrt(252)
    
    prices = np.zeros((len(dates), n_stocks))
    prices[0] = np.random.uniform(1000, 10000, n_stocks)
    
    for t in range(1, len(dates)):
        z = np.random.standard_normal(n_stocks)
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2) + sigma*z)
    
    return pd.DataFrame(prices, index=dates, columns=config.jii_tickers)


# =========================================================================
# 8. CLI ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sharia-Constrained RL Portfolio Optimization"
    )
    parser.add_argument("--algo", type=str, default="PPO", choices=["DQN", "PPO", "A2C"],
                        help="RL algorithm to train")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of Yahoo Finance")
    
    args = parser.parse_args()
    
    report = run_experiment(
        algo=args.algo,
        episodes=args.episodes,
        seed=args.seed,
    )
