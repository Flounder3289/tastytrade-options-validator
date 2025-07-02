   # enhanced_streamlit_app.py - Streamlit app with TastyTrade and OpenRouter integration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import requests

# Import our enhanced modules
try:
    from advanced_analytics import (
        AdvancedOptionsAnalyzer, AdvancedAnalysisResult, 
        format_advanced_results, get_historical_data
    )
    from langgraph_integration import run_langgraph_analysis, LANGGRAPH_AVAILABLE
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    st.error(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Import TastyTrade client
try:
    from tastytrade_client import TastyTradeClient
    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False
    st.warning("TastyTrade client not available. Create tastytrade_client.py file.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="TastyTrade Advanced Options Validator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .advanced-metric {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #667eea;
    }
    .kelly-score {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    .z-score-positive { color: #27ae60; }
    .z-score-negative { color: #e74c3c; }
    .confidence-indicator {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        color: white;
    }
    .conf-high { background-color: #27ae60; }
    .conf-medium { background-color: #f39c12; }
    .conf-low { background-color: #e74c3c; }
    .langgraph-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .monte-carlo-viz {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_api_credentials():
    """Get API credentials from various sources (secrets, env vars, or user input)"""
    credentials = {
        'tastytrade': {},
        'openrouter': {},
        'anthropic': {}
    }
    
    # Try Streamlit secrets first
    try:
        credentials['tastytrade'] = {
            'username': st.secrets.get("tastytrade", {}).get("username"),
            'password': st.secrets.get("tastytrade", {}).get("password")
        }
        credentials['openrouter']['api_key'] = st.secrets.get("openrouter", {}).get("api_key")
        credentials['anthropic']['api_key'] = st.secrets.get("anthropic", {}).get("api_key")
    except:
        pass
    # Initialize Anthropic API
anthropic_key = credentials['anthropic']['api_key']
if anthropic_key:
    apis['anthropic'] = {'api_key': anthropic_key}
    logger.info("Anthropic API key available")
    # Fallback to environment variables
    credentials['tastytrade']['username'] = credentials['tastytrade']['username'] or os.getenv("TASTYTRADE_USERNAME")
    credentials['tastytrade']['password'] = credentials['tastytrade']['password'] or os.getenv("TASTYTRADE_PASSWORD")
    credentials['openrouter']['api_key'] = credentials['openrouter']['api_key'] or os.getenv("OPENROUTER_API_KEY")
    credentials['anthropic']['api_key'] = credentials['anthropic']['api_key'] or os.getenv("ANTHROPIC_API_KEY")
    
    return credentials

def initialize_apis(credentials):
    """Initialize API clients with credentials"""
    apis = {}
    
    # Initialize TastyTrade API
    if TASTYTRADE_AVAILABLE:
        try:
            tt_username = credentials['tastytrade']['username'] or st.session_state.get('tastytrade_username')
            tt_password = credentials['tastytrade']['password'] or st.session_state.get('tastytrade_password')
            
            if tt_username and tt_password:
                tastytrade_client = TastyTradeClient()
                if tastytrade_client.authenticate(tt_username, tt_password):
                    apis['tastytrade'] = tastytrade_client
                    logger.info("TastyTrade API initialized successfully")
                else:
                    st.sidebar.error("❌ TastyTrade authentication failed")
            else:
                logger.warning("TastyTrade credentials not available")
        except Exception as e:
            st.sidebar.error(f"TastyTrade API initialization failed: {e}")
            logger.error(f"TastyTrade API error: {e}")
    
    # Initialize OpenRouter API (simple implementation)
    try:
        openrouter_key = credentials['openrouter']['api_key'] or st.session_state.get('openrouter_api_key')
        if openrouter_key:
            apis['openrouter'] = {'api_key': openrouter_key}
            logger.info("OpenRouter API key available")
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
    
    return apis

def setup_api_sidebar():
    """Setup API configuration in sidebar"""
    st.sidebar.header("🔐 API Configuration")
    
    credentials = get_api_credentials()
    
    # Check if credentials are available
    tastytrade_configured = all([
        credentials['tastytrade']['username'],
        credentials['tastytrade']['password']
    ])
    
    openrouter_configured = bool(credentials['openrouter']['api_key'])
    
    # Status indicators
    st.sidebar.subheader("📊 API Status")
    
    if 'apis' in st.session_state and st.session_state.apis.get('tastytrade'):
        st.sidebar.success("✅ TastyTrade: Connected")
    elif tastytrade_configured:
        st.sidebar.warning("🔄 TastyTrade: Configured, not connected")
    else:
        st.sidebar.error("❌ TastyTrade: Not configured")
    
    if openrouter_configured:
        st.sidebar.success("✅ OpenRouter: Configured")
    else:
        st.sidebar.error("❌ OpenRouter: Not configured")
    
    # Manual credential input if not configured
    if not tastytrade_configured and TASTYTRADE_AVAILABLE:
        st.sidebar.subheader("🔧 TastyTrade Login")
        
        with st.sidebar.form("tastytrade_login"):
            st.warning("⚠️ Only enter credentials if you trust this environment")
            
            tt_username = st.text_input("TastyTrade Username", 
                                      value=credentials['tastytrade']['username'] or "")
            tt_password = st.text_input("TastyTrade Password", 
                                      type="password")
            
            login_submitted = st.form_submit_button("🔐 Login to TastyTrade")
            
            if login_submitted and tt_username and tt_password:
                with st.spinner("Authenticating with TastyTrade..."):
                    tastytrade_client = TastyTradeClient()
                    if tastytrade_client.authenticate(tt_username, tt_password):
                        st.session_state.apis = st.session_state.get('apis', {})
                        st.session_state.apis['tastytrade'] = tastytrade_client
                        st.success("✅ TastyTrade login successful!")
                        st.rerun()
                    else:
                        st.error("❌ TastyTrade login failed")
    
    if not openrouter_configured:
        st.sidebar.subheader("🤖 OpenRouter API")
        or_api_key = st.sidebar.text_input("OpenRouter API Key", 
                                         type="password",
                                         key="or_api_key")
        
        if or_api_key:
            st.session_state.openrouter_api_key = or_api_key
    
    return credentials

def initialize_session_state():
    """Initialize session state with advanced features and API credentials"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'advanced_analyzer' not in st.session_state and ADVANCED_FEATURES_AVAILABLE:
        st.session_state.advanced_analyzer = AdvancedOptionsAnalyzer()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'langgraph_results' not in st.session_state:
        st.session_state.langgraph_results = None
    
    # Initialize API credentials
    if 'credentials' not in st.session_state:
        st.session_state.credentials = get_api_credentials()
    if 'apis' not in st.session_state:
        st.session_state.apis = {}

def show_advanced_metrics(advanced_results: AdvancedAnalysisResult):
    """Display advanced analytics metrics"""
    
    st.subheader("🧮 Advanced Analytics")
    
    # Kelly Criterion metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        kelly_color = "conf-high" if advanced_results.kelly_score >= 15 else "conf-medium" if advanced_results.kelly_score >= 8 else "conf-low"
        st.markdown(f'''
        <div class="advanced-metric">
            <div style="font-size: 0.9rem; color: #666;">Kelly Score</div>
            <div class="kelly-score">{advanced_results.kelly_score:.1f}/100</div>
            <div style="font-size: 0.8rem; color: #888;">Kelly Fraction: {advanced_results.kelly_fraction:.1%}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        z_color = "z-score-positive" if advanced_results.z_score > 0 else "z-score-negative"
        st.markdown(f'''
        <div class="advanced-metric">
            <div style="font-size: 0.9rem; color: #666;">Z-Score</div>
            <div class="{z_color}" style="font-size: 1.5rem; font-weight: bold;">{advanced_results.z_score:.2f}</div>
            <div style="font-size: 0.8rem; color: #888;">{advanced_results.z_score_interpretation}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="advanced-metric">
            <div style="font-size: 0.9rem; color: #666;">Black-Scholes POP</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{advanced_results.black_scholes_pop:.1%}</div>
            <div style="font-size: 0.8rem; color: #888;">Theoretical Model</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="advanced-metric">
            <div style="font-size: 0.9rem; color: #666;">Monte Carlo POP</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{advanced_results.monte_carlo_pop:.1%}</div>
            <div style="font-size: 0.8rem; color: #888;">Simulation Based</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Risk metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        expected_val = advanced_results.risk_metrics.get('expected_value', 0)
        ev_color = "green" if expected_val > 0 else "red"
        st.metric("Expected Value", f"${expected_val:.2f}", 
                 delta=f"{'Profitable' if expected_val > 0 else 'Loss Expected'}")
    
    with col2:
        var_95 = advanced_results.risk_metrics.get('var_95', 0)
        st.metric("VaR (95%)", f"${var_95:.2f}", 
                 help="Value at Risk - 95% confidence level")
    
    with col3:
        sharpe = advanced_results.risk_metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", 
                 help="Risk-adjusted return metric")
    
    with col4:
        recommended_size = advanced_results.recommended_position_size
        st.metric("Recommended Size", f"${recommended_size:,.0f}", 
                 help="Kelly-based position sizing")

def create_monte_carlo_visualization(profit_distribution: List[float]) -> go.Figure:
    """Create Monte Carlo profit distribution visualization"""
    
    if not profit_distribution:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Profit Distribution', 'Cumulative Distribution', 
                       'Box Plot', 'Probability Cone'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    profits = np.array(profit_distribution)
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=profits, nbinsx=50, name="Profit Distribution",
                    marker_color='lightblue', opacity=0.7),
        row=1, col=1
    )
    
    # Cumulative distribution
    sorted_profits = np.sort(profits)
    cumulative_prob = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
    
    fig.add_trace(
        go.Scatter(x=sorted_profits, y=cumulative_prob, mode='lines',
                  name="Cumulative Probability", line=dict(color='blue')),
        row=1, col=2
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=profits, name="Profit Range", marker_color='green'),
        row=2, col=1
    )
    
    # Probability cone (percentiles over time)
    percentiles = [5, 25, 50, 75, 95]
    perc_values = np.percentile(profits, percentiles)
    
    fig.add_trace(
        go.Scatter(x=percentiles, y=perc_values, mode='lines+markers',
                  name="Percentiles", line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Monte Carlo Analysis Results",
        showlegend=False
    )
    
    # Add profit/loss line
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Break-even", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    
    return fig

def show_langgraph_results(langgraph_results: Dict):
    """Display LangGraph workflow results"""
    
    if not langgraph_results.get('langgraph_enabled', False):
        st.warning("🤖 LangGraph analysis not available. Install langgraph package for advanced AI workflows.")
        return
    
    st.markdown('<div class="langgraph-section">', unsafe_allow_html=True)
    st.subheader("🧠 LangGraph AI Workflow Results")
    
    analysis_results = langgraph_results.get('analysis_results', {})
    final_rec = analysis_results.get('final_recommendation', {})
    
    # Main recommendation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recommendation = final_rec.get('overall_recommendation', 'UNKNOWN')
        confidence = final_rec.get('confidence_level', 5)
        
        rec_color = {
            'STRONG_BUY': 'conf-high',
            'BUY': 'conf-medium', 
            'HOLD': 'conf-medium',
            'AVOID': 'conf-low'
        }.get(recommendation, 'conf-medium')
        
        st.markdown(f'''
        <div class="confidence-indicator {rec_color}">
            {recommendation}
        </div>
        <div style="margin-top: 0.5rem;">Confidence: {confidence}/10</div>
        ''', unsafe_allow_html=True)
    
    with col2:
        composite_score = final_rec.get('composite_score', 5)
        st.metric("Composite Score", f"{composite_score:.1f}/10")
    
    with col3:
        individual_scores = final_rec.get('individual_scores', {})
        st.write("**Component Scores:**")
        for component, score in individual_scores.items():
            st.write(f"• {component.title()}: {score}/10")
    
    # Detailed analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["Market Analysis", "Risk Assessment", "Sentiment", "Validation"])
    
    with tab1:
        market_analysis = analysis_results.get('market_analysis', {})
        st.write("**Volatility Environment:**", market_analysis.get('volatility_environment', 'Unknown'))
        st.write("**Timing Analysis:**", market_analysis.get('timing_analysis', 'Unknown'))
        if 'sector_factors' in market_analysis:
            st.write("**Sector Factors:**", market_analysis['sector_factors'])
    
    with tab2:
        risk_assessment = analysis_results.get('risk_assessment', {})
        st.write("**Risk Level:**", risk_assessment.get('risk_level', 'Unknown'))
        
        risk_factors = risk_assessment.get('risk_factors', [])
        if risk_factors:
            st.write("**Key Risk Factors:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        
        if 'position_sizing' in risk_assessment:
            st.write("**Position Sizing:**", risk_assessment['position_sizing'])
    
    with tab3:
        sentiment_analysis = analysis_results.get('sentiment_analysis', {})
        st.write("**Market Sentiment:**", sentiment_analysis.get('market_sentiment', 'Unknown'))
        st.write("**Volatility Sentiment:**", sentiment_analysis.get('volatility_sentiment', 'Unknown'))
        
        if 'contrarian_signals' in sentiment_analysis:
            st.write("**Contrarian Signals:**", sentiment_analysis['contrarian_signals'])
    
    with tab4:
        validation = langgraph_results.get('validation', {})
        validation_passed = validation.get('validation_passed', True)
        
        if validation_passed:
            st.success("✅ Analysis passed validation checks")
        else:
            st.warning("⚠️ Some validation concerns identified")
        
        critical_issues = validation.get('critical_issues', [])
        if critical_issues:
            st.write("**Critical Issues:**")
            for issue in critical_issues:
                st.error(f"• {issue}")
        
        quality_score = validation.get('final_quality_score', 7)
        st.metric("Analysis Quality Score", f"{quality_score}/10")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_advanced_validation_tab():
    """Enhanced validation tab with live TastyTrade data and all advanced features"""
    
    st.header("🚀 Advanced Options Validator")
    
    # Enhanced input form
    with st.form("advanced_spread_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Basic Configuration")
            symbol = st.text_input("Symbol", value="SPY")
            spread_type = st.selectbox("Spread Type", ["PUT", "CALL"])
            
            # Live price from TastyTrade
            current_price = 450.0  # Default
            if st.session_state.get('apis', {}).get('tastytrade'):
                col_price, col_button = st.columns([2, 1])
                with col_button:
                    if st.form_submit_button("📡 Get Live Price", type="secondary"):
                        try:
                            api = st.session_state.apis['tastytrade']
                            live_price = api.get_current_price(symbol)
                            if live_price:
                                current_price = live_price
                                st.success(f"${live_price:.2f}")
                            else:
                                st.error("Price not found")
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with col_price:
                    current_price = st.number_input("Current Price", 
                                                  min_value=0.0, 
                                                  value=current_price, 
                                                  step=0.01)
            else:
                current_price = st.number_input("Current Price", 
                                              min_value=0.0, 
                                              value=450.0, 
                                              step=0.01)
                if TASTYTRADE_AVAILABLE:
                    st.info("💡 Connect TastyTrade for live prices")
            
        with col2:
            st.subheader("🎯 Strike Configuration")
            
            # Options chain from TastyTrade
            if st.session_state.get('apis', {}).get('tastytrade'):
                if st.form_submit_button("📊 Load Options Chain", type="secondary"):
                    try:
                        api = st.session_state.apis['tastytrade']
                        options_chain = api.get_options_chain(symbol)
                        if options_chain:
                            st.session_state.options_chain = options_chain
                            st.success("Options chain loaded")
                        else:
                            st.error("Failed to load options chain")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            short_strike = st.number_input("Short Strike", min_value=0.0, value=440.0, step=0.5)
            long_strike = st.number_input("Long Strike", min_value=0.0, value=435.0, step=0.5)
            credit = st.number_input("Credit", min_value=0.0, value=1.25, step=0.01)
            
        with col3:
            st.subheader("⏰ Timing & Risk")
            days_to_exp = st.number_input("Days to Expiration", min_value=1, max_value=365, value=21)
            implied_vol = st.slider("Implied Volatility", min_value=0.0, max_value=1.0, value=0.16, step=0.01)
            risk_free_rate = st.slider("Risk-Free Rate", min_value=0.0, max_value=0.10, value=0.05, step=0.001)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_monte_carlo = st.checkbox("Enable Monte Carlo Analysis", value=True)
                num_simulations = st.number_input("Monte Carlo Simulations", 
                                                min_value=1000, max_value=50000, value=10000, step=1000)
                use_live_data = st.checkbox("Use Live Market Data", 
                                          value=bool(st.session_state.get('apis', {}).get('tastytrade')),
                                          disabled=not bool(st.session_state.get('apis', {}).get('tastytrade')))
                
            with col2:
                enable_ai_analysis = st.checkbox("Enable AI Analysis", 
                                                value=bool(st.session_state.get('apis', {}).get('openrouter')),
                                                disabled=not bool(st.session_state.get('apis', {}).get('openrouter')))
                enable_langgraph = st.checkbox("Enable LangGraph AI Workflow", 
                                              value=LANGGRAPH_AVAILABLE,
                                              disabled=not LANGGRAPH_AVAILABLE)
                account_balance = st.number_input("Account Balance (for Kelly sizing)", 
                                                min_value=1000, value=100000, step=1000)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Run Advanced Analysis", type="primary")
    
    # Display current options chain if available
    if st.session_state.get('options_chain'):
        with st.expander("📊 Current Options Chain"):
            st.json(st.session_state.options_chain)
    
    # Quick metrics display
    if submitted or st.session_state.analysis_results:
        strike_width = abs(short_strike - long_strike)
        credit_ratio = credit / strike_width if strike_width > 0 else 0
        max_loss = strike_width - credit
        risk_reward = credit / max_loss if max_loss > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strike Width", f"${strike_width:.2f}")
        with col2:
            st.metric("Credit Ratio", f"{credit_ratio:.1%}")
        with col3:
            st.metric("Max Loss", f"${max_loss:.2f}")
        with col4:
            st.metric("Risk/Reward", f"{risk_reward:.2f}")
    
    # Run analysis if form submitted
    if submitted and ADVANCED_FEATURES_AVAILABLE:
        spread_data = {
            'symbol': symbol,
            'spread_type': spread_type,
            'current_price': current_price,
            'short_strike': short_strike,
            'long_strike': long_strike,
            'credit': credit,
            'days_to_exp': days_to_exp,
            'implied_volatility': implied_vol,
            'risk_free_rate': risk_free_rate,
            'account_balance': account_balance,
            'use_live_data': use_live_data,
            'enable_ai_analysis': enable_ai_analysis
        }
        
        with st.spinner("🧮 Running advanced analytics... This may take a moment."):
            try:
                # Run enhanced analysis with API integration
                run_enhanced_analysis(spread_data)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                logger.error(f"Advanced analysis error: {e}")
    
    # Display results
    if st.session_state.analysis_results:
        show_advanced_metrics(st.session_state.analysis_results)
        
        # Monte Carlo visualization
        st.subheader("📊 Monte Carlo Analysis")
        profit_dist = st.session_state.analysis_results.monte_carlo_profit_dist
        
        if profit_dist:
            fig = create_monte_carlo_visualization(profit_dist)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            profits = np.array(profit_dist)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean P&L", f"${np.mean(profits):.2f}")
            with col2:
                st.metric("Std Dev", f"${np.std(profits):.2f}")
            with col3:
                win_rate = np.sum(profits > 0) / len(profits)
                st.metric("Win Rate", f"{win_rate:.1%}")
            with col4:
                st.metric("Max Drawdown", f"${np.min(profits):.2f}")
        
        # Black-Scholes details
        st.subheader("⚡ Black-Scholes Analysis")
        bs_details = st.session_state.analysis_results.black_scholes_details
        
        if bs_details and 'error' not in bs_details:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Theoretical Credit", f"${bs_details.get('theoretical_credit', 0):.2f}")
                st.metric("Breakeven Price", f"${bs_details.get('breakeven_price', 0):.2f}")
            
            with col2:
                net_greeks = bs_details.get('net_greeks', {})
                st.metric("Net Delta", f"{net_greeks.get('delta', 0):.3f}")
                st.metric("Net Gamma", f"{net_greeks.get('gamma', 0):.3f}")
            
            with col3:
                st.metric("Net Theta", f"${net_greeks.get('theta', 0):.2f}")
                credit_accuracy = bs_details.get('credit_accuracy', 0)
                st.metric("Model Accuracy", f"{(1-credit_accuracy):.1%}")
    
    # LangGraph results
    if st.session_state.langgraph_results:
        show_langgraph_results(st.session_state.langgraph_results)
       # LangGraph results
    if st.session_state.langgraph_results:
        show_langgraph_results(st.session_state.langgraph_results)
        
    # Add debug display
    st.markdown("---")
    st.subheader("🔍 Debug Information")
    display_langgraph_debug()

# REPLACE the entire run_enhanced_analysis function (lines 675-761) with this:

def run_enhanced_analysis(spread_data):
    """Run enhanced analysis with API integration including LangGraph debug"""
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Get historical data
        if spread_data.get('use_live_data') and st.session_state.get('apis', {}).get('tastytrade'):
            historical_data = None
        else:
            historical_data = loop.run_until_complete(
                get_historical_data(spread_data['symbol'], None)
            )
        
        # Run comprehensive analysis
        analysis_results = loop.run_until_complete(
            st.session_state.advanced_analyzer.comprehensive_analysis(
                spread_data, historical_data
            )
        )
        
        # ===== DEBUG CODE START =====
        # Log analysis completion
        logger.info(f"Analysis completed. Results type: {type(analysis_results)}")
        logger.info(f"Analysis results attributes: {dir(analysis_results)}")
        
        # Check if LangGraph is available
        try:
            from langgraph_integration import LANGGRAPH_AVAILABLE
            logger.info(f"LangGraph available: {LANGGRAPH_AVAILABLE}")
        except ImportError:
            logger.info("LangGraph integration not imported")
            LANGGRAPH_AVAILABLE = False
        
        # Store any AI/LangGraph results in session state for debugging
        if hasattr(analysis_results, 'ai_analysis'):
            logger.info("AI analysis found in results")
            st.session_state.langgraph_results = analysis_results.ai_analysis
        elif hasattr(analysis_results, 'langgraph_analysis'):
            logger.info("LangGraph analysis found in results")
            st.session_state.langgraph_results = analysis_results.langgraph_analysis
        else:
            logger.info("No AI/LangGraph analysis found in results")
            # Store the entire results for debugging
            st.session_state.debug_analysis_results = {
                'type': str(type(analysis_results)),
                'attributes': [attr for attr in dir(analysis_results) if not attr.startswith('_')],
                'has_ai': hasattr(analysis_results, 'ai_analysis'),
                'has_langgraph': hasattr(analysis_results, 'langgraph_analysis')
            }
        # ===== DEBUG CODE END =====
        
        st.session_state.analysis_results = analysis_results
        
        # AI Analysis with OpenRouter
        if spread_data.get('enable_ai_analysis') and st.session_state.get('apis', {}).get('openrouter'):
            with st.spinner("🤖 Running AI analysis..."):
                ai_analysis = call_openrouter_api(spread_data, st.session_state.apis['openrouter']['api_key'])
                st.session_state.ai_analysis = ai_analysis
        
        # LangGraph analysis with Anthropic
        if spread_data.get('enable_langgraph', False) and LANGGRAPH_AVAILABLE:
            anthropic_key = st.session_state.credentials['anthropic']['api_key']
            if anthropic_key:
                with st.spinner("🧠 Running LangGraph AI workflow..."):
                    market_context = {
                        'vix': 16.5,  # Would normally fetch real VIX
                        'volume_ratio': 1.2,
                        'trend': 'neutral'
                    }
                    
                    langgraph_results = loop.run_until_complete(
                        run_langgraph_analysis(
                            spread_data,
                            anthropic_key,
                            market_context
                        )
                    )
                    
                    st.session_state.langgraph_analysis = langgraph_results
                    logger.info(f"LangGraph analysis completed: {langgraph_results.get('langgraph_enabled', False)}")
            else:
                st.warning("Anthropic API key required for LangGraph analysis")
        finally:
        loop.close()
       
                    # Market Analysis
                    if 'market_analysis' in results:
                        st.subheader("📊 Market Analysis")
                        market = results['market_analysis']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Volatility Environment:** {market.get('volatility_environment', 'Unknown')}")
                            st.write(f"**Timing Analysis:** {market.get('timing_analysis', 'Unknown')}")
                        with col2:
                            st.write(f"**Sector Factors:** {market.get('sector_factors', 'Unknown')}")
                            if 'risk_factors' in market:
                                st.write("**Risk Factors:**")
                                for risk in market['risk_factors']:
                                    st.write(f"• {risk}")
                    
                    # Risk Assessment
                    if 'risk_assessment' in results:
                        st.subheader("⚠️ Risk Assessment")
                        risk = results['risk_assessment']
                        st.write(f"**Risk Level:** {risk.get('risk_level', 'Unknown')}")
                        st.write(f"**Position Sizing:** {risk.get('position_sizing', 'Unknown')}")
                        st.write(f"**Exit Strategy:** {risk.get('exit_strategy', 'Unknown')}")
                    
                    # Final Recommendation
                    if 'final_recommendation' in results:
                        st.subheader("✅ AI Recommendation")
                        rec = results['final_recommendation']
                        
                        # Color code the recommendation
                        rec_color = {
                            'STRONG_BUY': 'green',
                            'BUY': 'blue',
                            'HOLD': 'orange',
                            'AVOID': 'red'
                        }.get(rec.get('overall_recommendation', 'HOLD'), 'gray')
                        
                        st.markdown(f"<h3 style='color: {rec_color};'>{rec.get('overall_recommendation', 'UNKNOWN')}</h3>", 
                                  unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence Level", f"{rec.get('confidence_level', 0)}/10")
                        with col2:
                            st.metric("Composite Score", f"{rec.get('composite_score', 0)}/10")
                        
                        if 'reasoning' in rec:
                            st.write("**Key Reasoning:**")
                            for reason in rec['reasoning']:
                                st.write(f"• {reason}")
            else:
                st.warning("⚠️ LangGraph not enabled - showing basic analysis")
                st.json(langgraph_data)

# ALSO, in show_advanced_validation_tab function, after line 672 (after showing langgraph results), ADD:

        # Add debug display
        st.markdown("---")
        st.subheader("🔍 Debug Information")
        display_langgraph_debug()
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        st.error(f"Analysis error: {str(e)}")
        return None
    finally:
        loop.close()
        
        # Run comprehensive analysis
        advanced_results = loop.run_until_complete(
            st.session_state.advanced_analyzer.comprehensive_analysis(
                spread_data, historical_data
            )
        )
        
        st.session_state.analysis_results = advanced_results
        
        # AI Analysis with OpenRouter
        if spread_data.get('enable_ai_analysis') and st.session_state.get('apis', {}).get('openrouter'):
            with st.spinner("🤖 Running AI analysis..."):
                # Simple OpenRouter API call
                ai_analysis = call_openrouter_api(spread_data, st.session_state.apis['openrouter']['api_key'])
                st.session_state.ai_analysis = ai_analysis
        
        # LangGraph analysis
        if spread_data.get('enable_langgraph', False) and LANGGRAPH_AVAILABLE:
            anthropic_key = st.session_state.credentials['anthropic']['api_key']
            if anthropic_key:
                with st.spinner("🧠 Running LangGraph AI workflow..."):
                    market_context = {'vix': 18.5, 'volume_ratio': 1.2, 'trend': 'Neutral'}
                    langgraph_results = loop.run_until_complete(
                        run_langgraph_analysis(spread_data, anthropic_key, market_context)
                    )
                    st.session_state.langgraph_results = langgraph_results
            else:
                st.warning("Anthropic API key required for LangGraph analysis")
    
    finally:
        loop.close()

def call_openrouter_api(spread_data, api_key):
    """Simple OpenRouter API call for options analysis"""
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Analyze this options spread:
        Symbol: {spread_data['symbol']}
        Type: {spread_data['spread_type']} spread
        Current Price: ${spread_data['current_price']}
        Short Strike: ${spread_data['short_strike']}
        Long Strike: ${spread_data['long_strike']}
        Credit: ${spread_data['credit']}
        Days to Expiration: {spread_data['days_to_exp']}
        
        Provide a brief analysis of the risk/reward and probability of success.
        """
        
        data = {
            "model": "anthropic/claude-3-haiku",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code}"
            
    except Exception as e:
        return f"Error calling OpenRouter API: {e}"

def show_api_management_tab():
    """API management and testing tab"""
    st.header("⚙️ API Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔌 TastyTrade API")
        
        if st.session_state.get('apis', {}).get('tastytrade'):
            st.success("✅ TastyTrade API Connected")
            
            if st.button("Test TastyTrade Connection"):
                with st.spinner("Testing TastyTrade connection..."):
                    try:
                        api = st.session_state.apis['tastytrade']
                        if api.is_authenticated():
                            st.success("✅ TastyTrade connection successful!")
                        else:
                            st.error("❌ TastyTrade connection failed")
                    except Exception as e:
                        st.error(f"❌ Connection error: {e}")
            
            # Show sample data
            if st.button("Fetch Sample Data"):
                with st.spinner("Fetching data from TastyTrade..."):
                    try:
                        api = st.session_state.apis['tastytrade']
                        sample_price = api.get_current_price("SPY")
                        
                        if sample_price:
                            st.json({"SPY_price": sample_price})
                        else:
                            st.warning("No data returned")
                    except Exception as e:
                        st.error(f"Error fetching data: {e}")
        else:
            st.warning("❌ TastyTrade API not configured")
            st.info("Please configure TastyTrade credentials in the sidebar")
    
    with col2:
        st.subheader("🤖 OpenRouter AI API")
        
        if st.session_state.get('apis', {}).get('openrouter'):
            st.success("✅ OpenRouter API Connected")
            
            if st.button("Test OpenRouter Connection"):
                with st.spinner("Testing OpenRouter connection..."):
                    try:
                        test_response = call_openrouter_api({
                            'symbol': 'SPY',
                            'spread_type': 'PUT',
                            'current_price': 450,
                            'short_strike': 440,
                            'long_strike': 435,
                            'credit': 1.25,
                            'days_to_exp': 21
                        }, st.session_state.apis['openrouter']['api_key'])
                        
                        st.write("**AI Response:**")
                        st.write(test_response)
                    except Exception as e:
                        st.error(f"Error in AI analysis: {e}")
        else:
            st.warning("❌ OpenRouter API not configured")
            st.info("Please configure OpenRouter API key in the sidebar")

def show_model_comparison_tab():
    """Compare different analysis models"""
    st.header("📊 Model Comparison")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # POP comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Probability of Profit Comparison")
            
            pop_data = {
                'Model': ['Black-Scholes', 'Monte Carlo', 'User Input'],
                'POP': [
                    results.black_scholes_pop,
                    results.monte_carlo_pop,
                    0.78  # Would come from user input
                ]
            }
            
            fig = px.bar(pop_data, x='Model', y='POP', 
                        title="POP Model Comparison",
                        color='POP', color_continuous_scale='RdYlGn')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Risk Metrics Comparison")
            
            # Create radar chart for risk metrics
            categories = ['Expected Value', 'Sharpe Ratio', 'Kelly Score', 'Z-Score']
            
            # Normalize values for radar chart
            values = [
                max(0, min(100, results.risk_metrics.get('expected_value', 0) * 10 + 50)),
                max(0, min(100, results.risk_metrics.get('sharpe_ratio', 0) * 20 + 50)),
                max(0, min(100, results.kelly_score)),
                max(0, min(100, abs(results.z_score) * 20 + 50))
            ]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],  # Close the loop
                theta=categories + [categories[0]],
                fill='toself',
                name='Risk Profile'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Risk Profile Radar"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run an analysis first to see model comparisons")

def show_performance_analytics_tab():
    """Show performance analytics and backtesting"""
    st.header("📈 Performance Analytics")
    
    # Mock historical performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate mock performance data
    np.random.seed(42)
    returns = np.random.normal(0.02, 0.15, len(dates))  # 2% mean return, 15% volatility
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    performance_df = pd.DataFrame({
        'Date': dates,
        'Daily_Return': returns,
        'Cumulative_Return': cumulative_returns,
        'Kelly_Score': np.random.uniform(5, 25, len(dates)),
        'Win_Rate': np.random.uniform(0.6, 0.8, len(dates))
    })
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Cumulative Returns")
        fig = px.line(performance_df, x='Date', y='Cumulative_Return',
                     title="Strategy Performance Over Time")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Kelly Score Evolution")
        fig = px.line(performance_df, x='Date', y='Kelly_Score',
                     title="Kelly Score Over Time")
        fig.add_hline(y=15, line_dash="dash", line_color="green", 
                     annotation_text="Strong Buy Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = cumulative_returns[-1]
        st.metric("Total Return", f"{total_return:.1%}")
    
    with col2:
        volatility = np.std(returns) * np.sqrt(252)
        st.metric("Annualized Volatility", f"{volatility:.1%}")
    
    with col3:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")

def show_research_tools_tab():
    """Advanced research and analysis tools"""
    st.header("🔬 Research Tools")
    
    # Option pricing model comparison
    st.subheader("🧮 Option Pricing Model Sandbox")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Parameters**")
        S = st.number_input("Stock Price", value=100.0, step=1.0)
        K = st.number_input("Strike Price", value=100.0, step=1.0)
        T = st.number_input("Time to Expiration (years)", value=0.25, step=0.01)
        r = st.number_input("Risk-Free Rate", value=0.05, step=0.01)
        sigma = st.number_input("Volatility", value=0.20, step=0.01)
    
    with col2:
        if st.button("Calculate Option Prices"):
            try:
                from advanced_analytics import BlackScholesAnalyzer
                
                bs = BlackScholesAnalyzer()
                
                call_price = bs.calculate_option_price(S, K, T, r, sigma, 'call')
                put_price = bs.calculate_option_price(S, K, T, r, sigma, 'put')
                
                greeks_call = bs.calculate_greeks(S, K, T, r, sigma, 'call')
                greeks_put = bs.calculate_greeks(S, K, T, r, sigma, 'put')
                
                st.write("**Call Option**")
                st.write(f"Price: ${call_price:.2f}")
                st.write(f"Delta: {greeks_call['delta']:.3f}")
                st.write(f"Gamma: {greeks_call['gamma']:.3f}")
                st.write(f"Theta: ${greeks_call['theta']:.2f}")
                
                st.write("**Put Option**")
                st.write(f"Price: ${put_price:.2f}")
                st.write(f"Delta: {greeks_put['delta']:.3f}")
                st.write(f"Gamma: {greeks_put['gamma']:.3f}")
                st.write(f"Theta: ${greeks_put['theta']:.2f}")
            except Exception as e:
                st.error(f"Error calculating prices: {e}")

def main():
    """Enhanced main application with TastyTrade and OpenRouter integration"""
    
    initialize_session_state()
    
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>🚀 TastyTrade Advanced Options Validator</h1>
        <p>Complete options analysis with live TastyTrade data, Kelly Criterion, Z-Score, Monte Carlo, Black-Scholes, and AI workflows</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Feature availability check
    if not ADVANCED_FEATURES_AVAILABLE:
        st.error("⚠️ Advanced features not available. Please ensure all required modules are installed.")
        st.stop()
    
    # Setup API configuration in sidebar
    credentials = setup_api_sidebar()
    
    # Initialize APIs if not already done
    if 'apis' not in st.session_state or not st.session_state.apis:
        st.session_state.apis = initialize_apis(credentials)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Advanced Configuration")
        
        # Feature toggles
        st.subheader("🎛️ Analysis Features")
        
        features = {
            "Kelly Criterion": st.checkbox("Kelly Criterion Position Sizing", value=True),
            "Monte Carlo": st.checkbox("Monte Carlo Simulation", value=True),
            "Black-Scholes": st.checkbox("Black-Scholes with Z-Score", value=True),
            "TastyTrade Data": st.checkbox("Live TastyTrade Data", 
                                         value=bool(st.session_state.get('apis', {}).get('tastytrade')),
                                         disabled=not bool(st.session_state.get('apis', {}).get('tastytrade'))),
            "OpenRouter AI": st.checkbox("OpenRouter AI Analysis", 
                                        value=bool(st.session_state.get('apis', {}).get('openrouter')),
                                        disabled=not bool(st.session_state.get('apis', {}).get('openrouter'))),
            "LangGraph": st.checkbox("LangGraph AI Workflow", value=LANGGRAPH_AVAILABLE, 
                                   disabled=not LANGGRAPH_AVAILABLE)
        }
        
        # Store features in session state
        st.session_state.enabled_features = features
        
        # Analysis parameters
        st.subheader("📊 Analysis Parameters")
        monte_carlo_sims = st.slider("Monte Carlo Simulations", 1000, 50000, 10000, step=1000)
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)
        
        # Model parameters
        st.subheader("🎯 Model Parameters")
        volatility_model = st.selectbox("Volatility Model", 
                                       ["Historical", "Implied", "GARCH", "Stochastic"])
        price_model = st.selectbox("Price Model", 
                                  ["Black-Scholes", "Heston", "Jump Diffusion"])
        
        # Display feature status
        st.subheader("📊 Feature Status")
        
        status_items = [
            ("Kelly Criterion", "✅ Available"),
            ("Z-Score Analysis", "✅ Available"), 
            ("Monte Carlo", "✅ Available"),
            ("Black-Scholes", "✅ Available"),
            ("TastyTrade", "✅ Available" if TASTYTRADE_AVAILABLE else "❌ Not Available"),
            ("LangGraph", "✅ Available" if LANGGRAPH_AVAILABLE else "❌ Not Available")
        ]
        
        for feature, status in status_items:
            st.text(f"{feature}: {status}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Advanced Analysis", 
        "📊 Model Comparison", 
        "📈 Performance Analytics", 
        "🔬 Research Tools",
        "⚙️ API Management"
    ])
    
    with tab1:
        show_advanced_validation_tab()
    
    with tab2:
        show_model_comparison_tab()
    
    with tab3:
        show_performance_analytics_tab()
    
    with tab4:
        show_research_tools_tab()
    
    with tab5:
        show_api_management_tab()

if __name__ == "__main__":
    main()
