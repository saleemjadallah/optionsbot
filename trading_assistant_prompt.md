# Multi-Model AI Trading Assistant - Implementation Prompt

## Overview
Build an intelligent AI chat interface integrated into the options trading bot that automatically routes queries to the optimal AI model (Claude, Perplexity, OpenAI) based on query type and context. The assistant should provide real-time support for trading operations, data analysis, strategy recommendations, and system configuration.

---

## Core Requirements

### 1. Multi-Model Architecture

```python
# Target file structure
options_trader/
├── ai_assistant/
│   ├── __init__.py
│   ├── model_router.py          # Intelligent query routing
│   ├── model_clients/
│   │   ├── __init__.py
│   │   ├── claude_client.py     # Anthropic Claude integration
│   │   ├── perplexity_client.py # Perplexity API integration
│   │   ├── openai_client.py     # OpenAI GPT integration
│   ├── context_manager.py       # Trading context aggregation
│   ├── prompt_templates.py      # Domain-specific prompts
│   ├── conversation_history.py  # Session management
│   └── tools/
│       ├── data_interpreter.py  # Real-time data analysis
│       ├── strategy_advisor.py  # Strategy recommendations
│       ├── risk_analyzer.py     # Risk analysis tools
│       └── system_config.py     # Configuration helper
├── ui/
│   └── chat_interface.py        # Streamlit chat UI
```

### 2. Model Selection Logic

The router should automatically select models based on:

**Claude Sonnet 4.5** - Primary for:
- Complex strategy analysis and recommendations
- Multi-step reasoning about trading decisions
- Risk management interpretation
- Code generation for custom indicators/strategies
- Long-form explanations of trading concepts
- Portfolio optimization suggestions
- Backtesting result interpretation

**Perplexity** - Primary for:
- Real-time market news and events
- Current options market conditions
- Latest regulatory changes
- Broker-specific information
- Recent market sentiment analysis
- Breaking news that might affect positions
- Competitor/peer analysis

**OpenAI GPT-4** - Primary for:
- Quick data calculations and transformations
- SQL query generation for historical data
- Simple parameter adjustments
- Configuration file modifications
- Routine Q&A about bot settings
- Fast responses for operational queries
- Data formatting and export tasks

---

## Implementation Specifications

### Phase 1: Model Client Implementations

#### 1.1 Claude Client (`model_clients/claude_client.py`)

```python
"""
Claude client for complex reasoning and strategy analysis.
Uses Anthropic's API with streaming support.
"""

import anthropic
import os
from typing import Optional, Dict, List, AsyncGenerator
import asyncio

class ClaudeClient:
    """
    Anthropic Claude client optimized for trading strategy analysis.
    
    Features:
    - Streaming responses for real-time feedback
    - Extended context for comprehensive analysis
    - Tool use for data access
    - Conversation history management
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"
        self.max_tokens = 8000
        
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        trading_context: Dict,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Send chat request to Claude with trading context.
        
        Args:
            messages: Conversation history
            system_prompt: System instructions with trading expertise
            trading_context: Current portfolio, positions, market data
            stream: Enable streaming responses
        
        Yields:
            Response chunks if streaming, otherwise full response
        """
        
        # Inject trading context into system prompt
        enhanced_system = self._build_system_prompt(system_prompt, trading_context)
        
        try:
            if stream:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=enhanced_system,
                    messages=messages,
                    temperature=0.7
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
            else:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=enhanced_system,
                    messages=messages,
                    temperature=0.7
                )
                yield response.content[0].text
                
        except Exception as e:
            yield f"Error communicating with Claude: {str(e)}"
    
    def _build_system_prompt(self, base_prompt: str, context: Dict) -> str:
        """Enhance system prompt with current trading context."""
        
        context_str = f"""
Current Trading Context:
- Portfolio Value: ${context.get('portfolio_value', 0):,.2f}
- Available Capital: ${context.get('available_capital', 0):,.2f}
- Active Positions: {context.get('position_count', 0)}
- Portfolio Delta: {context.get('delta', 0):.2f}
- Portfolio Gamma: {context.get('gamma', 0):.4f}
- Current VaR (95%): ${context.get('var', 0):,.2f}
- Risk Level: {context.get('risk_level', 'moderate')}
- Market Regime: {context.get('market_regime', 'normal')}

Recent Performance:
- Today's P&L: ${context.get('daily_pnl', 0):,.2f}
- Week's P&L: ${context.get('weekly_pnl', 0):,.2f}
- Month's P&L: ${context.get('monthly_pnl', 0):,.2f}
- Win Rate: {context.get('win_rate', 0):.1f}%
"""
        
        return f"{base_prompt}\n\n{context_str}"
    
    async def analyze_strategy(
        self,
        strategy_name: str,
        market_data: Dict,
        historical_performance: Dict
    ) -> str:
        """
        Specialized method for deep strategy analysis.
        
        Returns comprehensive analysis including:
        - Current market fit
        - Expected performance
        - Risk assessment
        - Parameter optimization suggestions
        """
        
        prompt = f"""Analyze the {strategy_name} strategy given:

Market Data:
{self._format_market_data(market_data)}

Historical Performance:
{self._format_performance(historical_performance)}

Provide:
1. Current market environment assessment
2. Strategy suitability score (1-10)
3. Expected performance over next 30 days
4. Key risks and mitigation strategies
5. Recommended parameter adjustments
6. Position sizing recommendation based on current regime
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        full_response = ""
        async for chunk in self.chat(messages, self._get_strategy_system_prompt(), {}):
            full_response += chunk
        
        return full_response
    
    def _get_strategy_system_prompt(self) -> str:
        return """You are an expert quantitative trading analyst specializing in options strategies.
Your role is to provide actionable, data-driven insights for options trading decisions.

Core Expertise:
- Advanced options pricing models (Black-Scholes, Merton Jump Diffusion, Heston, SABR)
- Greeks analysis and portfolio risk management
- Volatility modeling and trading
- Market regime detection
- Statistical arbitrage

Communication Style:
- Direct and actionable recommendations
- Quantitative reasoning with specific numbers
- Clear risk/reward assessments
- Concrete parameter suggestions
- Prioritize practical implementation

Always consider:
1. Current market volatility regime
2. Portfolio Greeks exposure
3. Risk-adjusted returns
4. Transaction costs and slippage
5. Position sizing based on Kelly Criterion
"""
    
    def _format_market_data(self, data: Dict) -> str:
        """Format market data for prompt."""
        return "\n".join([f"- {k}: {v}" for k, v in data.items()])
    
    def _format_performance(self, perf: Dict) -> str:
        """Format performance metrics for prompt."""
        return "\n".join([f"- {k}: {v}" for k, v in perf.items()])
```

#### 1.2 Perplexity Client (`model_clients/perplexity_client.py`)

```python
"""
Perplexity client for real-time market intelligence and news.
"""

import aiohttp
import os
from typing import Optional, Dict, List
from datetime import datetime

class PerplexityClient:
    """
    Perplexity API client for real-time market research and news.
    
    Features:
    - Real-time web search
    - Current market conditions
    - News and sentiment analysis
    - Regulatory updates
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"
        
    async def search_market_news(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Search for relevant market news and analysis.
        
        Args:
            query: User's question or search topic
            symbols: List of ticker symbols to focus on
            context: Additional trading context
        
        Returns:
            Dictionary with response and sources
        """
        
        # Enhance query with trading context
        enhanced_query = self._enhance_query(query, symbols, context)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._get_market_research_prompt()
                },
                {
                    "role": "user",
                    "content": enhanced_query
                }
            ],
            "temperature": 0.2,
            "return_citations": True,
            "return_related_questions": True
        }
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                self.base_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._format_response(data)
                else:
                    return {
                        "answer": f"Error: {response.status}",
                        "sources": [],
                        "error": True
                    }
    
    async def get_options_market_update(
        self,
        symbols: List[str],
        include_iv_analysis: bool = True
    ) -> Dict:
        """
        Get current options market conditions for specified symbols.
        """
        
        query = f"""Current options market analysis for {', '.join(symbols)}:
        
1. Recent unusual options activity
2. Implied volatility trends (past 5 days)
3. Notable large trades or sweeps
4. Any relevant news or catalysts
5. Market maker positioning insights
{' 6. IV rank and percentile analysis' if include_iv_analysis else ''}

Focus on actionable information for options traders."""
        
        return await self.search_market_news(query, symbols)
    
    async def check_market_regime(self) -> Dict:
        """
        Assess current market regime and volatility environment.
        """
        
        query = """Current market regime analysis:
        
1. VIX level and recent trend
2. Market correlation levels
3. Risk appetite indicators (HYG/LQD spreads, etc.)
4. Major economic events this week
5. Fed policy stance and upcoming decisions
6. Geopolitical risks affecting volatility

Provide specific numbers and recent changes."""
        
        return await self.search_market_news(query)
    
    def _enhance_query(
        self,
        query: str,
        symbols: Optional[List[str]],
        context: Optional[Dict]
    ) -> str:
        """Add relevant context to search query."""
        
        enhancements = []
        
        if symbols:
            enhancements.append(f"Focus on: {', '.join(symbols)}")
        
        if context and context.get('positions'):
            enhancements.append(
                f"Relevant to current positions in {', '.join(context['positions'])}"
            )
        
        enhancements.append(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
        
        enhanced = query
        if enhancements:
            enhanced += "\n\nContext:\n" + "\n".join(enhancements)
        
        return enhanced
    
    def _get_market_research_prompt(self) -> str:
        return """You are a real-time market intelligence analyst specializing in options markets.

Your role:
- Provide up-to-date market information with specific data points
- Focus on actionable intelligence for options traders
- Include relevant statistics (IV levels, volume, price movements)
- Cite credible sources (Bloomberg, Reuters, exchange data)
- Highlight time-sensitive information

Response format:
1. Key findings (bullet points with specific numbers)
2. Market implications for options traders
3. Relevant timeframes (immediate, this week, this month)
4. Risk factors to monitor

Always:
- Include specific numbers and percentages
- Mention data timestamps/recency
- Distinguish between facts and analysis
- Flag any major uncertainties
"""
    
    def _format_response(self, data: Dict) -> Dict:
        """Format Perplexity response for consistent interface."""
        
        message = data.get('choices', [{}])[0].get('message', {})
        
        return {
            "answer": message.get('content', ''),
            "sources": data.get('citations', []),
            "related_questions": data.get('related_questions', []),
            "timestamp": datetime.now().isoformat(),
            "error": False
        }
```

#### 1.3 OpenAI Client (`model_clients/openai_client.py`)

```python
"""
OpenAI GPT client for quick calculations and routine operations.
"""

import openai
import os
from typing import Optional, Dict, List, Any
import json

class OpenAIClient:
    """
    OpenAI GPT client for fast operations and calculations.
    
    Features:
    - Function calling for tool integration
    - Quick responses for operational queries
    - Data transformations and calculations
    - Configuration assistance
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key
        self.model = "gpt-4-turbo-preview"
        
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.3
    ) -> Dict:
        """
        Send chat request with optional function calling.
        
        Args:
            messages: Conversation history
            tools: Available functions for tool use
            temperature: Response randomness (lower = more deterministic)
        
        Returns:
            Response with potential function calls
        """
        
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            response = await openai.ChatCompletion.acreate(**params)
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "tool_calls": None,
                "finish_reason": "error"
            }
    
    async def calculate_greeks(
        self,
        option_params: Dict[str, Any]
    ) -> Dict:
        """
        Quick Greeks calculation using GPT with function calling.
        """
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_option_greeks",
                    "description": "Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot_price": {"type": "number"},
                            "strike_price": {"type": "number"},
                            "time_to_expiry": {"type": "number"},
                            "risk_free_rate": {"type": "number"},
                            "volatility": {"type": "number"},
                            "option_type": {"type": "string", "enum": ["call", "put"]}
                        },
                        "required": ["spot_price", "strike_price", "time_to_expiry", 
                                   "risk_free_rate", "volatility", "option_type"]
                    }
                }
            }
        ]
        
        messages = [
            {
                "role": "system",
                "content": "You are a financial calculator assistant. Calculate option Greeks when requested."
            },
            {
                "role": "user",
                "content": f"Calculate Greeks for: {json.dumps(option_params)}"
            }
        ]
        
        response = await self.chat(messages, tools=tools)
        
        if response["tool_calls"]:
            # Execute the actual calculation
            from models.greeks import GreeksCalculator
            calc = GreeksCalculator()
            
            args = json.loads(response["tool_calls"][0].function.arguments)
            greeks = calc.calculate(**args)
            
            return greeks
        
        return {}
    
    async def generate_sql_query(
        self,
        natural_language_query: str,
        database_schema: Dict
    ) -> str:
        """
        Generate SQL query from natural language using GPT.
        """
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a SQL expert. Generate PostgreSQL queries based on natural language.
                
Database schema:
{json.dumps(database_schema, indent=2)}

Generate only the SQL query, no explanations."""
            },
            {
                "role": "user",
                "content": natural_language_query
            }
        ]
        
        response = await self.chat(messages, temperature=0)
        return response["content"]
    
    async def format_data(
        self,
        data: Any,
        target_format: str,
        options: Optional[Dict] = None
    ) -> str:
        """
        Transform data into requested format (CSV, JSON, markdown table, etc.).
        """
        
        messages = [
            {
                "role": "system",
                "content": f"Convert the provided data to {target_format} format."
            },
            {
                "role": "user",
                "content": f"Data: {json.dumps(data)}\n\nOptions: {json.dumps(options or {})}"
            }
        ]
        
        response = await self.chat(messages, temperature=0)
        return response["content"]
```

---

### Phase 2: Intelligent Query Router

#### 2.1 Model Router (`model_router.py`)

```python
"""
Intelligent routing system that selects the optimal AI model for each query.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

class QueryType(Enum):
    """Classification of query types."""
    STRATEGY_ANALYSIS = "strategy_analysis"
    MARKET_NEWS = "market_news"
    DATA_CALCULATION = "data_calculation"
    RISK_ASSESSMENT = "risk_assessment"
    SYSTEM_CONFIG = "system_config"
    GENERAL_QUESTION = "general_question"
    BACKTESTING = "backtesting"
    PARAMETER_TUNING = "parameter_tuning"

@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence: float
    preferred_model: str
    reasoning: str
    keywords: List[str]

class ModelRouter:
    """
    Routes queries to the optimal AI model based on query characteristics.
    
    Routing Logic:
    - Claude: Complex reasoning, strategy analysis, risk assessment
    - Perplexity: Real-time market data, news, current events
    - OpenAI: Quick calculations, data formatting, routine operations
    """
    
    def __init__(self):
        # Keyword patterns for classification
        self.patterns = {
            QueryType.MARKET_NEWS: [
                r'\b(news|headline|breaking|latest|current|recent|today)\b',
                r'\b(market|sentiment|catalyst|event)\b',
                r'\b(happening|announced|reported)\b',
                r'\b(IV|implied volatility).*(trend|change|spike)\b'
            ],
            QueryType.DATA_CALCULATION: [
                r'\b(calculate|compute|what is|how much)\b',
                r'\b(delta|gamma|theta|vega|rho)\b',
                r'\b(price|value|profit|loss|return)\b',
                r'\b(convert|transform|format)\b',
                r'\b(sql|query|database)\b'
            ],
            QueryType.STRATEGY_ANALYSIS: [
                r'\b(strategy|approach|should I|recommend)\b',
                r'\b(analyze|evaluate|assess|review)\b',
                r'\b(performance|backtest|optimize)\b',
                r'\b(dispersion|gamma scalp|iron condor|straddle)\b',
                r'\b(why|explain|how does|reasoning)\b'
            ],
            QueryType.RISK_ASSESSMENT: [
                r'\b(risk|var|exposure|hedge)\b',
                r'\b(portfolio|position size|allocation)\b',
                r'\b(safe|dangerous|protect|loss)\b',
                r'\b(regime|volatility|correlation)\b'
            ],
            QueryType.SYSTEM_CONFIG: [
                r'\b(config|setting|parameter|adjust)\b',
                r'\b(change|modify|update|set)\b',
                r'\b(threshold|limit|max|min)\b',
                r'\b(enable|disable|turn on|turn off)\b'
            ],
            QueryType.BACKTESTING: [
                r'\b(backtest|historical|past performance)\b',
                r'\b(simulate|test|would have)\b',
                r'\b(from.*to|over the past|last.*month)\b'
            ],
            QueryType.PARAMETER_TUNING: [
                r'\b(optimize|tune|calibrate|fit)\b',
                r'\b(parameter|coefficient|weight)\b',
                r'\b(improve|better|enhance)\b'
            ]
        }
        
        # Model preference matrix
        self.model_preferences = {
            QueryType.MARKET_NEWS: "perplexity",
            QueryType.DATA_CALCULATION: "openai",
            QueryType.STRATEGY_ANALYSIS: "claude",
            QueryType.RISK_ASSESSMENT: "claude",
            QueryType.SYSTEM_CONFIG: "openai",
            QueryType.GENERAL_QUESTION: "claude",
            QueryType.BACKTESTING: "claude",
            QueryType.PARAMETER_TUNING: "claude"
        }
        
    def classify_query(self, query: str, context: Optional[Dict] = None) -> QueryClassification:
        """
        Classify query and determine optimal model.
        
        Args:
            query: User's input question/request
            context: Additional context (recent messages, trading state)
        
        Returns:
            Classification with model recommendation
        """
        
        query_lower = query.lower()
        scores = {qt: 0.0 for qt in QueryType}
        matched_keywords = {qt: [] for qt in QueryType}
        
        # Score each query type based on pattern matches
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    scores[query_type] += len(matches)
                    matched_keywords[query_type].extend(matches)
        
        # Apply context-based adjustments
        if context:
            scores = self._adjust_scores_by_context(scores, context)
        
        # Determine best match
        if max(scores.values()) == 0:
            # No clear match, default to general question
            best_type = QueryType.GENERAL_QUESTION
            confidence = 0.3
        else:
            best_type = max(scores, key=scores.get)
            total_score = sum(scores.values())
            confidence = scores[best_type] / total_score if total_score > 0 else 0
        
        preferred_model = self.model_preferences[best_type]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_type, matched_keywords[best_type], confidence)
        
        return QueryClassification(
            query_type=best_type,
            confidence=confidence,
            preferred_model=preferred_model,
            reasoning=reasoning,
            keywords=matched_keywords[best_type]
        )
    
    def _adjust_scores_by_context(self, scores: Dict, context: Dict) -> Dict:
        """Adjust classification scores based on context."""
        
        # If user just asked about news, likely following up
        if context.get('last_query_type') == QueryType.MARKET_NEWS:
            scores[QueryType.MARKET_NEWS] *= 1.5
        
        # If system is currently executing trades, prioritize operational queries
        if context.get('active_trading'):
            scores[QueryType.DATA_CALCULATION] *= 1.3
            scores[QueryType.RISK_ASSESSMENT] *= 1.3
        
        # If user is in configuration mode
        if context.get('config_mode'):
            scores[QueryType.SYSTEM_CONFIG] *= 2.0
        
        return scores
    
    def _generate_reasoning(self, query_type: QueryType, keywords: List[str], confidence: float) -> str:
        """Generate human-readable reasoning for model selection."""
        
        reasons = {
            QueryType.MARKET_NEWS: "Query appears to request current market information or news",
            QueryType.DATA_CALCULATION: "Query involves numerical calculations or data transformations",
            QueryType.STRATEGY_ANALYSIS: "Query requires strategic thinking or complex analysis",
            QueryType.RISK_ASSESSMENT: "Query focuses on risk evaluation or portfolio protection",
            QueryType.SYSTEM_CONFIG: "Query relates to system settings or configuration changes",
            QueryType.GENERAL_QUESTION: "General query without specific domain focus",
            QueryType.BACKTESTING: "Query involves historical performance analysis",
            QueryType.PARAMETER_TUNING: "Query relates to optimization or parameter adjustment"
        }
        
        reasoning = reasons[query_type]
        
        if keywords:
            reasoning += f" (matched: {', '.join(set(keywords)[:3])})"
        
        reasoning += f" [confidence: {confidence:.0%}]"
        
        return reasoning
    
    async def route_query(
        self,
        query: str,
        context: Dict,
        claude_client,
        perplexity_client,
        openai_client,
        force_model: Optional[str] = None
    ) -> Tuple[str, QueryClassification]:
        """
        Route query to appropriate model and return response.
        
        Args:
            query: User's question
            context: Trading context and conversation history
            claude_client: Claude client instance
            perplexity_client: Perplexity client instance
            openai_client: OpenAI client instance
            force_model: Override automatic routing (for user preference)
        
        Returns:
            Tuple of (response, classification)
        """
        
        # Classify query
        classification = self.classify_query(query, context)
        
        # Determine which model to use
        if force_model:
            model = force_model
        else:
            model = classification.preferred_model
        
        # Route to appropriate model
        try:
            if model == "claude":
                response = await self._query_claude(
                    query, context, claude_client, classification
                )
            elif model == "perplexity":
                response = await self._query_perplexity(
                    query, context, perplexity_client, classification
                )
            elif model == "openai":
                response = await self._query_openai(
                    query, context, openai_client, classification
                )
            else:
                response = "Error: Unknown model specified"
        
        except Exception as e:
            # Fallback to Claude on error
            response = f"Error with {model}: {str(e)}\n\nFalling back to Claude..."
            response += await self._query_claude(query, context, claude_client, classification)
        
        return response, classification
    
    async def _query_claude(
        self,
        query: str,
        context: Dict,
        client,
        classification: QueryClassification
    ) -> str:
        """Execute query using Claude."""
        
        messages = self._build_messages(query, context)
        system_prompt = self._get_system_prompt("claude", classification)
        
        response = ""
        async for chunk in client.chat(messages, system_prompt, context):
            response += chunk
        
        return response
    
    async def _query_perplexity(
        self,
        query: str,
        context: Dict,
        client,
        classification: QueryClassification
    ) -> str:
        """Execute query using Perplexity."""
        
        symbols = context.get('active_symbols', [])
        result = await client.search_market_news(query, symbols, context)
        
        # Format response with sources
        response = result['answer']
        
        if result.get('sources'):
            response += "\n\n**Sources:**\n"
            for i, source in enumerate(result['sources'][:5], 1):
                response += f"{i}. {source}\n"
        
        return response
    
    async def _query_openai(
        self,
        query: str,
        context: Dict,
        client,
        classification: QueryClassification
    ) -> str:
        """Execute query using OpenAI."""
        
        messages = self._build_messages(query, context)
        result = await client.chat(messages)
        
        return result['content']
    
    def _build_messages(self, query: str, context: Dict) -> List[Dict]:
        """Build message list from conversation history."""
        
        messages = []
        
        # Add recent conversation history
        if context.get('conversation_history'):
            messages.extend(context['conversation_history'][-10:])  # Last 10 messages
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def _get_system_prompt(self, model: str, classification: QueryClassification) -> str:
        """Get appropriate system prompt based on model and query type."""
        
        base_prompts = {
            "claude": """You are an expert options trading assistant integrated into an algorithmic trading system.
Your role is to help traders make informed decisions by analyzing strategies, assessing risks, and providing actionable insights.

Available Information:
- Real-time portfolio data and positions
- Historical performance metrics
- Current market conditions
- Risk parameters and limits

Communication Style:
- Direct and actionable
- Use specific numbers and calculations
- Explain reasoning clearly
- Highlight risks and considerations
- Provide concrete recommendations when appropriate""",
            
            "openai": """You are a fast, efficient trading operations assistant.
Focus on quick calculations, data formatting, and routine tasks.
Be concise and precise in your responses."""
        }
        
        return base_prompts.get(model, base_prompts["claude"])
```

---

### Phase 3: Context Management

#### 3.1 Trading Context Manager (`context_manager.py`)

```python
"""
Aggregates and manages trading context for AI assistant.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

class TradingContextManager:
    """
    Manages comprehensive trading context for AI conversations.
    
    Aggregates:
    - Portfolio state (positions, Greeks, P&L)
    - Recent trades and performance
    - Market data and indicators
    - System status and alerts
    - User preferences and risk settings
    """
    
    def __init__(self, portfolio_manager, risk_manager, market_data_manager):
        self.portfolio = portfolio_manager
        self.risk = risk_manager
        self.market_data = market_data_manager
        self.conversation_history = []
        
    async def get_full_context(self) -> Dict:
        """
        Aggregate all relevant context for AI assistant.
        
        Returns comprehensive dictionary with:
        - Portfolio snapshot
        - Risk metrics
        - Recent activity
        - Market conditions
        - System status
        """
        
        context = {
            # Portfolio State
            "portfolio_value": self.portfolio.get_total_value(),
            "available_capital": self.portfolio.get_available_capital(),
            "position_count": len(self.portfolio.positions),
            "active_symbols": self.portfolio.get_active_symbols(),
            
            # Greeks
            "delta": self.portfolio.get_total_delta(),
            "gamma": self.portfolio.get_total_gamma(),
            "theta": self.portfolio.get_total_theta(),
            "vega": self.portfolio.get_total_vega(),
            
            # Risk Metrics
            "var": self.risk.calculate_var(0.95),
            "cvar": self.risk.calculate_cvar(0.95),
            "leverage": self.portfolio.get_leverage(),
            "risk_level": self.risk.current_risk_level,
            "market_regime": self.risk.current_regime,
            
            # Performance
            "daily_pnl": self.portfolio.get_pnl(timedelta(days=1)),
            "weekly_pnl": self.portfolio.get_pnl(timedelta(days=7)),
            "monthly_pnl": self.portfolio.get_pnl(timedelta(days=30)),
            "win_rate": self.portfolio.get_win_rate(),
            
            # Recent Activity
            "recent_trades": await self._get_recent_trades(),
            "recent_alerts": await self._get_recent_alerts(),
            
            # Market Data
            "vix_level": await self.market_data.get_vix(),
            "spy_price": await self.market_data.get_price("SPY"),
            "market_hours": self.market_data.is_market_open(),
            
            # System Status
            "active_strategies": self.portfolio.get_active_strategies(),
            "system_health": await self._check_system_health(),
            "last_update": datetime.now().isoformat(),
            
            # Conversation Context
            "conversation_history": self.conversation_history[-10:],
            "last_query_type": self.conversation_history[-1]['type'] if self.conversation_history else None
        }
        
        return context
    
    async def get_position_details(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get detailed information about current positions."""
        
        positions = self.portfolio.get_positions(symbol)
        
        details = []
        for pos in positions:
            details.append({
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "pnl": pos.calculate_pnl(),
                "pnl_percent": pos.calculate_pnl_percent(),
                "delta": pos.delta,
                "gamma": pos.gamma,
                "theta": pos.theta,
                "vega": pos.vega,
                "days_held": (datetime.now() - pos.entry_time).days,
                "strategy": pos.strategy_name
            })
        
        return details
    
    async def get_strategy_performance(self, strategy_name: Optional[str] = None) -> Dict:
        """Get performance metrics for specific strategy or all strategies."""
        
        if strategy_name:
            strategies = [strategy_name]
        else:
            strategies = self.portfolio.get_active_strategies()
        
        performance = {}
        
        for strat in strategies:
            trades = self.portfolio.get_trades_by_strategy(strat)
            
            if trades:
                performance[strat] = {
                    "total_trades": len(trades),
                    "winning_trades": len([t for t in trades if t.pnl > 0]),
                    "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades),
                    "avg_win": sum(t.pnl for t in trades if t.pnl > 0) / len([t for t in trades if t.pnl > 0]) if any(t.pnl > 0 for t in trades) else 0,
                    "avg_loss": sum(t.pnl for t in trades if t.pnl < 0) / len([t for t in trades if t.pnl < 0]) if any(t.pnl < 0 for t in trades) else 0,
                    "total_pnl": sum(t.pnl for t in trades),
                    "sharpe_ratio": self._calculate_sharpe([t.pnl for t in trades]),
                    "max_drawdown": self._calculate_max_drawdown([t.pnl for t in trades])
                }
        
        return performance
    
    def add_conversation_turn(self, query: str, response: str, query_type: str):
        """Record conversation turn for context."""
        
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "type": query_type
        })
        
        # Keep only last 50 turns
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    async def _get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Fetch recent trade history."""
        trades = self.portfolio.get_recent_trades(limit)
        return [self._format_trade(t) for t in trades]
    
    async def _get_recent_alerts(self, limit: int = 5) -> List[Dict]:
        """Fetch recent system alerts."""
        # Implementation depends on alerting system
        return []
    
    async def _check_system_health(self) -> Dict:
        """Check overall system health status."""
        return {
            "api_connected": True,  # Check Tastyworks connection
            "data_feed_active": True,  # Check market data
            "strategies_running": True,  # Check strategy execution
            "risk_limits_ok": True  # Check risk violations
        }
    
    def _format_trade(self, trade) -> Dict:
        """Format trade for context."""
        return {
            "symbol": trade.symbol,
            "action": trade.action,
            "quantity": trade.quantity,
            "price": trade.price,
            "pnl": trade.pnl,
            "timestamp": trade.timestamp.isoformat()
        }
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns:
            return 0
        return (pd.Series(returns).mean() / pd.Series(returns).std()) * (252 ** 0.5)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0
        cumulative = pd.Series(returns).cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

---

### Phase 4: Streamlit Chat Interface

#### 4.1 Chat UI (`ui/chat_interface.py`)

```python
"""
Streamlit-based chat interface for AI trading assistant.
"""

import streamlit as st
import asyncio
from datetime import datetime
from typing import Optional

# Import AI components
from ai_assistant.model_router import ModelRouter, QueryClassification
from ai_assistant.model_clients.claude_client import ClaudeClient
from ai_assistant.model_clients.perplexity_client import PerplexityClient
from ai_assistant.model_clients.openai_client import OpenAIClient
from ai_assistant.context_manager import TradingContextManager

class TradingChatInterface:
    """
    Main chat interface for trading assistant.
    
    Features:
    - Real-time responses with streaming
    - Model selection transparency
    - Context-aware conversations
    - Quick actions and shortcuts
    """
    
    def __init__(self):
        self.initialize_clients()
        self.initialize_session_state()
    
    def initialize_clients(self):
        """Initialize AI model clients."""
        
        if 'claude_client' not in st.session_state:
            st.session_state.claude_client = ClaudeClient()
        
        if 'perplexity_client' not in st.session_state:
            st.session_state.perplexity_client = PerplexityClient()
        
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = OpenAIClient()
        
        if 'model_router' not in st.session_state:
            st.session_state.model_router = ModelRouter()
        
        if 'context_manager' not in st.session_state:
            # Get instances from main app
            from main import bot
            st.session_state.context_manager = TradingContextManager(
                bot.portfolio_manager,
                bot.risk_manager,
                bot.market_data_manager
            )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'force_model' not in st.session_state:
            st.session_state.force_model = None
        
        if 'show_debug' not in st.session_state:
            st.session_state.show_debug = False
    
    def render(self):
        """Render the main chat interface."""
        
        st.title("🤖 AI Trading Assistant")
        
        # Sidebar configuration
        with st.sidebar:
            self.render_sidebar()
        
        # Main chat area
        self.render_chat_messages()
        
        # Input area
        self.render_input_area()
    
    def render_sidebar(self):
        """Render sidebar with settings and quick actions."""
        
        st.header("Settings")
        
        # Model override
        model_options = ["Auto", "Claude", "Perplexity", "OpenAI"]
        selected_model = st.selectbox(
            "Force Model",
            model_options,
            index=0,
            help="Override automatic model selection"
        )
        st.session_state.force_model = None if selected_model == "Auto" else selected_model.lower()
        
        # Debug mode
        st.session_state.show_debug = st.checkbox("Show Debug Info", value=False)
        
        st.divider()
        
        # Quick Actions
        st.header("Quick Actions")
        
        if st.button("📊 Portfolio Summary"):
            self.quick_action("Give me a comprehensive portfolio summary")
        
        if st.button("📰 Market Update"):
            self.quick_action("What's happening in the options market today?")
        
        if st.button("⚠️ Risk Check"):
            self.quick_action("Analyze my current risk exposure")
        
        if st.button("💡 Strategy Ideas"):
            self.quick_action("Based on current market conditions, suggest profitable strategies")
        
        if st.button("📈 Performance Review"):
            self.quick_action("Review my trading performance this month")
        
        st.divider()
        
        # Context Display
        st.header("Current Context")
        
        with st.expander("Portfolio State"):
            context = asyncio.run(st.session_state.context_manager.get_full_context())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Portfolio Value", f"${context['portfolio_value']:,.0f}")
                st.metric("Delta", f"{context['delta']:.2f}")
                st.metric("Risk Level", context['risk_level'].title())
            
            with col2:
                st.metric("Today's P&L", f"${context['daily_pnl']:,.0f}")
                st.metric("Gamma", f"{context['gamma']:.4f}")
                st.metric("Market Regime", context['market_regime'].title())
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    def render_chat_messages(self):
        """Render chat message history."""
        
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show model and classification info in debug mode
                    if st.session_state.show_debug and "classification" in message:
                        with st.expander("🔍 Debug Info"):
                            classification = message["classification"]
                            st.write(f"**Model Used:** {message['model']}")
                            st.write(f"**Query Type:** {classification.query_type.value}")
                            st.write(f"**Confidence:** {classification.confidence:.0%}")
                            st.write(f"**Reasoning:** {classification.reasoning}")
    
    def render_input_area(self):
        """Render input area for new messages."""
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about trading..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                # Show thinking indicator
                with st.spinner("Thinking..."):
                    response, classification, model_used = asyncio.run(
                        self.get_ai_response(prompt)
                    )
                
                # Display response
                response_placeholder.markdown(response)
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "classification": classification,
                    "model": model_used,
                    "timestamp": datetime.now().isoformat()
                })
    
    async def get_ai_response(self, query: str) -> tuple:
        """
        Get AI response for query.
        
        Returns:
            Tuple of (response, classification, model_used)
        """
        
        # Get current trading context
        context = await st.session_state.context_manager.get_full_context()
        
        # Route query to appropriate model
        response, classification = await st.session_state.model_router.route_query(
            query=query,
            context=context,
            claude_client=st.session_state.claude_client,
            perplexity_client=st.session_state.perplexity_client,
            openai_client=st.session_state.openai_client,
            force_model=st.session_state.force_model
        )
        
        model_used = st.session_state.force_model or classification.preferred_model
        
        # Update conversation history
        st.session_state.context_manager.add_conversation_turn(
            query, response, classification.query_type.value
        )
        
        return response, classification, model_used
    
    def quick_action(self, query: str):
        """Execute quick action by adding message to chat."""
        
        # Simulate user input
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Trigger rerun to process message
        st.rerun()

def main():
    """Main entry point for chat interface."""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI Trading Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize and render interface
    interface = TradingChatInterface()
    interface.render()

if __name__ == "__main__":
    main()
```

---

### Phase 5: Integration with Main Bot

#### 5.1 Integration Points

Add to `main.py`:

```python
# Add to imports
from ai_assistant.context_manager import TradingContextManager
from ui.chat_interface import TradingChatInterface

class OptionsTradingBot:
    def __init__(self):
        # ... existing initialization ...
        
        # Initialize AI assistant
        self.context_manager = TradingContextManager(
            self.portfolio_manager,
            self.risk_manager,
            self.market_data_manager
        )
    
    def launch_assistant_ui(self):
        """Launch AI assistant chat interface in separate process."""
        
        import subprocess
        subprocess.Popen([
            "streamlit", "run",
            "ui/chat_interface.py",
            "--server.port", "8501"
        ])
```

---

## Environment Variables

Create `.env` file:

```bash
# AI Model API Keys
ANTHROPIC_API_KEY=your_claude_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Tastyworks (existing)
TW_USERNAME=your_username
TW_PASSWORD=your_password
TW_ACCOUNT=your_account_number

# Database (existing)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=options_trader
DB_USER=trader
DB_PASSWORD=your_db_password
```

---

## Usage Examples

### Example 1: Strategy Analysis
```
User: "Should I continue running my gamma scalping strategy given the recent VIX spike?"

[System routes to Claude]