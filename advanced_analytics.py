# advanced_analytics.py - Advanced options analytics with Kelly Criterion, Z-Score, Monte Carlo
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class AdvancedAnalysisResult:
    """Container for advanced analysis results"""
    kelly_score: float
    kelly_fraction: float
    z_score: float
    z_score_interpretation: str
    black_scholes_pop: float
    monte_carlo_pop: float
    black_scholes_details: Dict[str, Any]
    monte_carlo_profit_dist: List[float]
    risk_metrics: Dict[str, float]
    recommended_position_size: float
    confidence_level: float

class BlackScholesAnalyzer:
    """Black-Scholes option pricing and Greeks calculator"""
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def calculate_option_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        try:
            d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
            
            if option_type.lower() == 'call':
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            
            return max(0, price)
        except Exception as e:
            logger.error(f"Error calculating option price: {e}")
            return 0.0
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)
            
            # Common Greeks
            gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
            
            if option_type.lower() == 'call':
                delta = stats.norm.cdf(d1)
                theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365
            else:  # put
                delta = stats.norm.cdf(d1) - 1
                theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

class MonteCarloSimulator:
    """Monte Carlo simulation for options strategies"""
    
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
    
    def simulate_price_paths(self, S0: float, T: float, r: float, sigma: float, 
                           num_paths: int = None) -> np.ndarray:
        """Simulate stock price paths using geometric Brownian motion"""
        if num_paths is None:
            num_paths = self.num_simulations
        
        dt = T / 252  # Daily steps
        num_steps = int(T * 252)
        
        # Generate random paths
        Z = np.random.standard_normal((num_steps, num_paths))
        
        # Calculate price paths
        price_paths = np.zeros((num_steps + 1, num_paths))
        price_paths[0] = S0
        
        for t in range(1, num_steps + 1):
            price_paths[t] = price_paths[t-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1]
            )
        
        return price_paths[-1]  # Return final prices
    
    def calculate_spread_profit(self, final_prices: np.ndarray, short_strike: float, 
                              long_strike: float, credit: float, spread_type: str) -> np.ndarray:
        """Calculate profit/loss for spread strategy"""
        profits = np.full_like(final_prices, credit)  # Start with credit received
        
        if spread_type.upper() == 'PUT':
            # Put spread
            short_assignment = np.maximum(short_strike - final_prices, 0)
            long_protection = np.maximum(long_strike - final_prices, 0)
            
            profits -= (short_assignment - long_protection)
        else:
            # Call spread
            short_assignment = np.maximum(final_prices - short_strike, 0)
            long_protection = np.maximum(final_prices - long_strike, 0)
            
            profits -= (short_assignment - long_protection)
        
        return profits
    
    def run_simulation(self, S0: float, T: float, r: float, sigma: float,
                      short_strike: float, long_strike: float, credit: float,
                      spread_type: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run complete Monte Carlo simulation"""
        # Simulate price paths
        final_prices = self.simulate_price_paths(S0, T, r, sigma)
        
        # Calculate profits
        profits = self.calculate_spread_profit(final_prices, short_strike, long_strike, credit, spread_type)
        
        # Calculate statistics
        stats_dict = {
            'mean_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'prob_profit': np.mean(profits > 0),
            'prob_max_loss': np.mean(profits <= -(abs(long_strike - short_strike) - credit)),
            'var_95': np.percentile(profits, 5),
            'var_99': np.percentile(profits, 1),
            'expected_return': np.mean(profits),
            'sharpe_ratio': np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
        }
        
        return profits, stats_dict

class KellyCriterionCalculator:
    """Kelly Criterion position sizing calculator"""
    
    @staticmethod
    def calculate_kelly_fraction(win_prob: float, win_amount: float, loss_amount: float) -> float:
        """Calculate Kelly fraction: f = (bp - q) / b"""
        try:
            if win_amount <= 0 or loss_amount <= 0:
                return 0.0
            
            b = win_amount / loss_amount  # Odds ratio
            p = win_prob  # Win probability
            q = 1 - p  # Loss probability
            
            kelly_f = (b * p - q) / b
            return max(0, min(kelly_f, 1.0))  # Constrain between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0
    
    @staticmethod
    def calculate_kelly_score(kelly_fraction: float) -> float:
        """Convert Kelly fraction to interpretable score (0-100)"""
        return min(100, kelly_fraction * 100 * 1.5)  # Scale for interpretation
    
    def calculate_position_size(self, kelly_fraction: float, account_balance: float,
                              max_risk_per_trade: float = 0.02) -> float:
        """Calculate recommended position size"""
        # Use fractional Kelly to reduce risk
        conservative_kelly = kelly_fraction * 0.25  # Use 25% of full Kelly
        
        # Apply maximum risk constraint
        risk_constrained = min(conservative_kelly, max_risk_per_trade)
        
        return account_balance * risk_constrained

class ZScoreAnalyzer:
    """Z-Score analysis for options strategies"""
    
    @staticmethod
    def calculate_z_score(actual_prob: float, expected_prob: float, 
                         sample_size: float = 10000) -> Tuple[float, str]:
        """Calculate Z-score for probability assessment"""
        try:
            # Standard error of proportion
            se = np.sqrt(expected_prob * (1 - expected_prob) / sample_size)
            
            if se == 0:
                return 0.0, "Insufficient data"
            
            z_score = (actual_prob - expected_prob) / se
            
            # Interpretation
            if abs(z_score) < 1.96:
                interpretation = "Not statistically significant"
            elif z_score > 1.96:
                interpretation = "Significantly higher than expected"
            else:
                interpretation = "Significantly lower than expected"
            
            return z_score, interpretation
        except Exception as e:
            logger.error(f"Error calculating Z-score: {e}")
            return 0.0, "Error in calculation"

class AdvancedOptionsAnalyzer:
    """Main class for advanced options analysis"""
    
    def __init__(self):
        self.bs_analyzer = BlackScholesAnalyzer()
        self.mc_simulator = MonteCarloSimulator()
        self.kelly_calculator = KellyCriterionCalculator()
        self.z_score_analyzer = ZScoreAnalyzer()
    
    async def comprehensive_analysis(self, spread_data: Dict, historical_data: Optional[pd.DataFrame] = None) -> AdvancedAnalysisResult:
        """Run comprehensive analysis combining all methods"""
        try:
            # Extract parameters
            S = spread_data['current_price']
            K_short = spread_data['short_strike']
            K_long = spread_data['long_strike']
            T = spread_data['days_to_exp'] / 365.0
            r = spread_data['risk_free_rate']
            sigma = spread_data['implied_volatility']
            credit = spread_data['credit']
            spread_type = spread_data['spread_type']
            account_balance = spread_data.get('account_balance', 100000)
            
            # Calculate max loss
            max_loss = abs(K_short - K_long) - credit
            
            # Black-Scholes Analysis
            bs_results = await self._black_scholes_analysis(S, K_short, K_long, T, r, sigma, spread_type)
            
            # Monte Carlo Analysis
            mc_profits, mc_stats = self.mc_simulator.run_simulation(
                S, T, r, sigma, K_short, K_long, credit, spread_type
            )
            
            # Kelly Criterion
            win_prob = mc_stats['prob_profit']
            kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
                win_prob, credit, max_loss
            )
            kelly_score = self.kelly_calculator.calculate_kelly_score(kelly_fraction)
            
            # Position sizing
            recommended_size = self.kelly_calculator.calculate_position_size(
                kelly_fraction, account_balance
            )
            
            # Z-Score Analysis
            z_score, z_interpretation = self.z_score_analyzer.calculate_z_score(
                mc_stats['prob_profit'], bs_results['pop']
            )
            
            # Risk metrics
            risk_metrics = {
                'expected_value': mc_stats['expected_return'],
                'var_95': mc_stats['var_95'],
                'sharpe_ratio': mc_stats['sharpe_ratio'],
                'max_loss': max_loss,
                'win_rate': win_prob
            }
            
            # Confidence level based on consistency between models
            model_agreement = 1 - abs(mc_stats['prob_profit'] - bs_results['pop'])
            confidence_level = min(10, max(1, model_agreement * 10))
            
            return AdvancedAnalysisResult(
                kelly_score=kelly_score,
                kelly_fraction=kelly_fraction,
                z_score=z_score,
                z_score_interpretation=z_interpretation,
                black_scholes_pop=bs_results['pop'],
                monte_carlo_pop=mc_stats['prob_profit'],
                black_scholes_details=bs_results,
                monte_carlo_profit_dist=mc_profits.tolist(),
                risk_metrics=risk_metrics,
                recommended_position_size=recommended_size,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # Return default results
            return AdvancedAnalysisResult(
                kelly_score=0.0,
                kelly_fraction=0.0,
                z_score=0.0,
                z_score_interpretation="Analysis failed",
                black_scholes_pop=0.0,
                monte_carlo_pop=0.0,
                black_scholes_details={},
                monte_carlo_profit_dist=[],
                risk_metrics={},
                recommended_position_size=0.0,
                confidence_level=1.0
            )
    
    async def _black_scholes_analysis(self, S: float, K_short: float, K_long: float,
                                    T: float, r: float, sigma: float, spread_type: str) -> Dict:
        """Perform Black-Scholes analysis for spread"""
        try:
            # Calculate option prices and Greeks
            if spread_type.upper() == 'PUT':
                short_put_price = self.bs_analyzer.calculate_option_price(S, K_short, T, r, sigma, 'put')
                long_put_price = self.bs_analyzer.calculate_option_price(S, K_long, T, r, sigma, 'put')
                
                short_greeks = self.bs_analyzer.calculate_greeks(S, K_short, T, r, sigma, 'put')
                long_greeks = self.bs_analyzer.calculate_greeks(S, K_long, T, r, sigma, 'put')
                
                theoretical_credit = short_put_price - long_put_price
                breakeven_price = K_short - theoretical_credit
                
                # POP for put spread (price stays above short strike)
                d1_short, d2_short = self.bs_analyzer.calculate_d1_d2(S, K_short, T, r, sigma)
                pop = stats.norm.cdf(d2_short)
                
            else:  # CALL spread
                short_call_price = self.bs_analyzer.calculate_option_price(S, K_short, T, r, sigma, 'call')
                long_call_price = self.bs_analyzer.calculate_option_price(S, K_long, T, r, sigma, 'call')
                
                short_greeks = self.bs_analyzer.calculate_greeks(S, K_short, T, r, sigma, 'call')
                long_greeks = self.bs_analyzer.calculate_greeks(S, K_long, T, r, sigma, 'call')
                
                theoretical_credit = short_call_price - long_call_price
                breakeven_price = K_short + theoretical_credit
                
                # POP for call spread (price stays below short strike)
                d1_short, d2_short = self.bs_analyzer.calculate_d1_d2(S, K_short, T, r, sigma)
                pop = stats.norm.cdf(-d2_short)
            
            # Net Greeks
            net_greeks = {
                'delta': short_greeks['delta'] - long_greeks['delta'],
                'gamma': short_greeks['gamma'] - long_greeks['gamma'],
                'theta': short_greeks['theta'] - long_greeks['theta'],
                'vega': short_greeks['vega'] - long_greeks['vega']
            }
            
            return {
                'pop': pop,
                'theoretical_credit': theoretical_credit,
                'breakeven_price': breakeven_price,
                'net_greeks': net_greeks,
                'short_greeks': short_greeks,
                'long_greeks': long_greeks
            }
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes analysis: {e}")
            return {'error': str(e), 'pop': 0.0}

async def get_historical_data(symbol: str, api_client=None) -> Optional[pd.DataFrame]:
    """Get historical data for volatility calculations"""
    try:
        if api_client:
            # Use API client if available
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
            
            historical_data = api_client.get_historical_data(symbol, start_date, end_date)
            if historical_data:
                df = pd.DataFrame(historical_data)
                return df
        
        # Fallback to yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        return hist
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return None

def format_advanced_results(results: AdvancedAnalysisResult) -> str:
    """Format results for display"""
    return f"""
    Advanced Analysis Results:
    
    Kelly Score: {results.kelly_score:.1f}/100
    Kelly Fraction: {results.kelly_fraction:.1%}
    Z-Score: {results.z_score:.2f} ({results.z_score_interpretation})
    
    Probability of Profit:
    - Black-Scholes: {results.black_scholes_pop:.1%}
    - Monte Carlo: {results.monte_carlo_pop:.1%}
    
    Risk Metrics:
    - Expected Value: ${results.risk_metrics.get('expected_value', 0):.2f}
    - VaR (95%): ${results.risk_metrics.get('var_95', 0):.2f}
    - Sharpe Ratio: {results.risk_metrics.get('sharpe_ratio', 0):.2f}
    
    Recommended Position Size: ${results.recommended_position_size:,.0f}
    Confidence Level: {results.confidence_level:.1f}/10
    """
