# langgraph_integration.py - LangGraph AI workflow integration
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check if LangGraph is available
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict
    import anthropic
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph available")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available - install with: pip install langgraph")

@dataclass
class AnalysisState:
    """State for LangGraph workflow"""
    spread_data: Dict[str, Any]
    market_context: Dict[str, Any]
    analysis_results: Dict[str, Any]
    validation: Dict[str, Any]
    messages: List[Dict[str, str]]

class OptionsAnalysisWorkflow:
    """LangGraph workflow for comprehensive options analysis"""
    
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        # Define the workflow state
        class WorkflowState(TypedDict):
            spread_data: Dict[str, Any]
            market_context: Dict[str, Any]
            analysis_results: Dict[str, Any]
            validation: Dict[str, Any]
            messages: List[Dict[str, str]]
        
        # Create workflow
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("market_analysis", self.market_analysis_node)
        workflow.add_node("risk_assessment", self.risk_assessment_node)
        workflow.add_node("sentiment_analysis", self.sentiment_analysis_node)
        workflow.add_node("strategy_validation", self.strategy_validation_node)
        workflow.add_node("final_recommendation", self.final_recommendation_node)
        
        # Add edges
        workflow.add_edge(START, "market_analysis")
        workflow.add_edge("market_analysis", "risk_assessment")
        workflow.add_edge("risk_assessment", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "strategy_validation")
        workflow.add_edge("strategy_validation", "final_recommendation")
        workflow.add_edge("final_recommendation", END)
        
        return workflow.compile()
    
    async def market_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions and volatility environment"""
        try:
            spread_data = state['spread_data']
            market_context = state['market_context']
            
            prompt = f"""
            Analyze the market environment for this options strategy:
            
            Symbol: {spread_data['symbol']}
            Current Price: ${spread_data['current_price']}
            Strategy: {spread_data['spread_type']} spread
            Short Strike: ${spread_data['short_strike']}
            Long Strike: ${spread_data['long_strike']}
            Days to Expiration: {spread_data['days_to_exp']}
            
            Market Context:
            VIX Level: {market_context.get('vix', 'Unknown')}
            Volume Ratio: {market_context.get('volume_ratio', 'Unknown')}
            Trend: {market_context.get('trend', 'Unknown')}
            
            Provide analysis on:
            1. Volatility environment (high/low/normal)
            2. Timing considerations
            3. Sector-specific factors
            4. Risk factors to monitor
            
            Format as JSON with keys: volatility_environment, timing_analysis, sector_factors, risk_factors
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            try:
                analysis = json.loads(response.content[0].text)
            except:
                analysis = {
                    "volatility_environment": "Analysis failed",
                    "timing_analysis": "Unable to determine",
                    "sector_factors": "Unknown",
                    "risk_factors": []
                }
            
            state['analysis_results']['market_analysis'] = analysis
            state['messages'].append({
                "node": "market_analysis",
                "result": "Market analysis completed"
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            state['analysis_results']['market_analysis'] = {"error": str(e)}
            return state
    
    async def risk_assessment_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk profile and position sizing"""
        try:
            spread_data = state['spread_data']
            market_analysis = state['analysis_results'].get('market_analysis', {})
            
            max_loss = abs(spread_data['short_strike'] - spread_data['long_strike']) - spread_data['credit']
            credit_ratio = spread_data['credit'] / abs(spread_data['short_strike'] - spread_data['long_strike'])
            
            prompt = f"""
            Assess the risk profile for this options strategy:
            
            Strategy Details:
            - Max Loss: ${max_loss:.2f}
            - Credit Received: ${spread_data['credit']:.2f}
            - Credit Ratio: {credit_ratio:.1%}
            - Time to Expiration: {spread_data['days_to_exp']} days
            
            Market Analysis Results:
            {json.dumps(market_analysis, indent=2)}
            
            Provide risk assessment including:
            1. Overall risk level (Low/Medium/High)
            2. Key risk factors
            3. Position sizing recommendations
            4. Exit strategy considerations
            
            Format as JSON with keys: risk_level, risk_factors, position_sizing, exit_strategy
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
                risk_assessment = json.loads(response.content[0].text)
            except:
                risk_assessment = {
                    "risk_level": "Medium",
                    "risk_factors": ["Analysis failed"],
                    "position_sizing": "Conservative sizing recommended",
                    "exit_strategy": "Monitor closely"
                }
            
            state['analysis_results']['risk_assessment'] = risk_assessment
            state['messages'].append({
                "node": "risk_assessment",
                "result": "Risk assessment completed"
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            state['analysis_results']['risk_assessment'] = {"error": str(e)}
            return state
    
    async def sentiment_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment and contrarian indicators"""
        try:
            spread_data = state['spread_data']
            market_context = state['market_context']
            
            prompt = f"""
            Analyze market sentiment for options strategy timing:
            
            Current Market Metrics:
            - VIX: {market_context.get('vix', 'Unknown')}
            - Volume Ratio: {market_context.get('volume_ratio', 'Unknown')}
            - Market Trend: {market_context.get('trend', 'Unknown')}
            
            Strategy: {spread_data['spread_type']} spread on {spread_data['symbol']}
            
            Analyze:
            1. Current market sentiment (bullish/bearish/neutral)
            2. Volatility sentiment (high/low relative to historical)
            3. Contrarian signals present
            4. Optimal timing considerations
            
            Format as JSON with keys: market_sentiment, volatility_sentiment, contrarian_signals, timing_score
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
                sentiment_analysis = json.loads(response.content[0].text)
            except:
                sentiment_analysis = {
                    "market_sentiment": "Neutral",
                    "volatility_sentiment": "Normal",
                    "contrarian_signals": "None detected",
                    "timing_score": 5
                }
            
            state['analysis_results']['sentiment_analysis'] = sentiment_analysis
            state['messages'].append({
                "node": "sentiment_analysis", 
                "result": "Sentiment analysis completed"
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            state['analysis_results']['sentiment_analysis'] = {"error": str(e)}
            return state
    
    async def strategy_validation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy against best practices"""
        try:
            spread_data = state['spread_data']
            analysis_results = state['analysis_results']
            
            # Validation checks
            validation_checks = []
            critical_issues = []
            
            # Credit ratio check
            strike_width = abs(spread_data['short_strike'] - spread_data['long_strike'])
            credit_ratio = spread_data['credit'] / strike_width if strike_width > 0 else 0
            
            if credit_ratio < 0.2:
                critical_issues.append("Credit ratio too low (< 20%)")
            elif credit_ratio > 0.5:
                validation_checks.append("Excellent credit ratio (> 50%)")
            else:
                validation_checks.append("Acceptable credit ratio")
            
            # Time decay check
            if spread_data['days_to_exp'] < 7:
                critical_issues.append("Very short time to expiration - high gamma risk")
            elif spread_data['days_to_exp'] < 21:
                validation_checks.append("Short DTE - monitor theta decay closely")
            
            # Distance from money check
            current_price = spread_data['current_price']
            short_strike = spread_data['short_strike']
            
            if spread_data['spread_type'].upper() == 'PUT':
                otm_amount = (current_price - short_strike) / current_price
            else:
                otm_amount = (short_strike - current_price) / current_price
            
            if otm_amount < 0.02:
                critical_issues.append("Too close to the money - high assignment risk")
            elif otm_amount > 0.15:
                validation_checks.append("Well out of the money - lower probability")
            
            # Calculate quality score
            quality_score = 10
            quality_score -= len(critical_issues) * 3
            quality_score -= max(0, len(validation_checks) - 2)
            quality_score = max(1, min(10, quality_score))
            
            validation = {
                "validation_passed": len(critical_issues) == 0,
                "validation_checks": validation_checks,
                "critical_issues": critical_issues,
                "credit_ratio": credit_ratio,
                "otm_percentage": otm_amount,
                "final_quality_score": quality_score
            }
            
            state['validation'] = validation
            state['messages'].append({
                "node": "strategy_validation",
                "result": f"Validation completed - Quality Score: {quality_score}/10"
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Strategy validation error: {e}")
            state['validation'] = {"error": str(e)}
            return state
    
    async def final_recommendation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final recommendation based on all analyses"""
        try:
            spread_data = state['spread_data']
            analysis_results = state['analysis_results']
            validation = state['validation']
            
            # Compile all analysis for final recommendation
            prompt = f"""
            Based on comprehensive analysis, provide final recommendation:
            
            Strategy: {spread_data['spread_type']} spread on {spread_data['symbol']}
            Credit: ${spread_data['credit']:.2f}
            Days to Expiration: {spread_data['days_to_exp']}
            
            Analysis Results:
            {json.dumps(analysis_results, indent=2)}
            
            Validation Results:
            {json.dumps(validation, indent=2)}
            
            Provide final recommendation with:
            1. Overall recommendation (STRONG_BUY/BUY/HOLD/AVOID)
            2. Confidence level (1-10)
            3. Composite score (1-10)
            4. Individual component scores for: market_timing, risk_reward, volatility_environment, strategy_setup
            5. Key reasoning points
            
            Format as JSON with keys: overall_recommendation, confidence_level, composite_score, individual_scores, reasoning
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
                final_rec = json.loads(response.content[0].text)
            except:
                final_rec = {
                    "overall_recommendation": "HOLD",
                    "confidence_level": 5,
                    "composite_score": 5,
                    "individual_scores": {
                        "market_timing": 5,
                        "risk_reward": 5,
                        "volatility_environment": 5,
                        "strategy_setup": 5
                    },
                    "reasoning": ["Analysis failed to parse properly"]
                }
            
            state['analysis_results']['final_recommendation'] = final_rec
            state['messages'].append({
                "node": "final_recommendation",
                "result": f"Final recommendation: {final_rec.get('overall_recommendation', 'UNKNOWN')}"
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Final recommendation error: {e}")
            state['analysis_results']['final_recommendation'] = {"error": str(e)}
            return state

class MockLangGraphWorkflow:
    """Mock workflow when LangGraph is not available"""
    
    def __init__(self, anthropic_api_key: str = None):
        self.client = None
        if anthropic_api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=anthropic_api_key)
            except ImportError:
                pass
    
    async def run_analysis(self, spread_data: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run simplified analysis without LangGraph"""
        
        # Simple rule-based analysis
        credit_ratio = spread_data['credit'] / abs(spread_data['short_strike'] - spread_data['long_strike'])
        days_to_exp = spread_data['days_to_exp']
        
        # Basic scoring
        scores = {
            "market_timing": 6,
            "risk_reward": min(10, max(1, credit_ratio * 20)),
            "volatility_environment": 5,
            "strategy_setup": 7 if days_to_exp > 14 else 4
        }
        
        composite_score = sum(scores.values()) / len(scores)
        
        if composite_score >= 7:
            recommendation = "BUY"
            confidence = 7
        elif composite_score >= 5:
            recommendation = "HOLD"
            confidence = 5
        else:
            recommendation = "AVOID"
            confidence = 6
        
        return {
            "langgraph_enabled": False,
            "analysis_results": {
                "market_analysis": {
                    "volatility_environment": "Normal",
                    "timing_analysis": "Adequate timing for strategy"
                },
                "risk_assessment": {
                    "risk_level": "Medium",
                    "risk_factors": ["Standard options risks apply"]
                },
                "sentiment_analysis": {
                    "market_sentiment": "Neutral",
                    "volatility_sentiment": "Normal"
                },
                "final_recommendation": {
                    "overall_recommendation": recommendation,
                    "confidence_level": confidence,
                    "composite_score": composite_score,
                    "individual_scores": scores,
                    "reasoning": ["Basic analysis - LangGraph not available"]
                }
            },
            "validation": {
                "validation_passed": True,
                "critical_issues": [],
                "final_quality_score": composite_score
            }
        }

async def run_langgraph_analysis(spread_data: Dict[str, Any], anthropic_api_key: str, 
                               market_context: Dict[str, Any]) -> Dict[str, Any]:
    """Run LangGraph analysis workflow"""
    
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, running mock analysis")
        mock_workflow = MockLangGraphWorkflow(anthropic_api_key)
        return await mock_workflow.run_analysis(spread_data, market_context)
    
    try:
        # Initialize workflow
        workflow = OptionsAnalysisWorkflow(anthropic_api_key)
        
        if not workflow.workflow:
            logger.error("Failed to initialize LangGraph workflow")
            mock_workflow = MockLangGraphWorkflow(anthropic_api_key)
            return await mock_workflow.run_analysis(spread_data, market_context)
        
        # Prepare initial state
        initial_state = {
            "spread_data": spread_data,
            "market_context": market_context,
            "analysis_results": {},
            "validation": {},
            "messages": []
        }
        
        # Run workflow
        result = await workflow.workflow.ainvoke(initial_state)
        
        # Format results
        return {
            "langgraph_enabled": True,
            "analysis_results": result.get("analysis_results", {}),
            "validation": result.get("validation", {}),
            "workflow_messages": result.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"LangGraph workflow error: {e}")
        
        # Fallback to mock analysis
        mock_workflow = MockLangGraphWorkflow(anthropic_api_key)
        return await mock_workflow.run_analysis(spread_data, market_context)
