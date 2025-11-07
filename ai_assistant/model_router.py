"""Query classification and routing logic for Jeffrey."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


class QueryType(enum.Enum):
    MARKET_NEWS = "market_news"
    DATA_CALCULATION = "data_calculation"
    STRATEGY_ANALYSIS = "strategy_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    SYSTEM_CONFIG = "system_config"
    GENERAL_QUESTION = "general_question"
    BACKTESTING = "backtesting"
    PARAMETER_TUNING = "parameter_tuning"


@dataclass
class QueryClassification:
    query_type: QueryType
    confidence: float
    preferred_model: str
    reasoning: str
    keywords: List[str]


class ModelRouter:
    def __init__(self) -> None:
        self.patterns: Dict[QueryType, List[str]] = {
            QueryType.MARKET_NEWS: [r"\b(news|headline|breaking|today|market)\b", r"\b(update|latest)\b"],
            QueryType.DATA_CALCULATION: [r"\b(calculate|calc|compute|what is)\b", r"\b(delta|gamma|theta|profit|loss)\b"],
            QueryType.STRATEGY_ANALYSIS: [r"\b(strategy|setup|structure|should i)\b", r"\b(analyze|assess|recommend)\b"],
            QueryType.RISK_ASSESSMENT: [r"\b(risk|var|hedge|exposure)\b", r"\b(safe|danger|limit)\b"],
            QueryType.SYSTEM_CONFIG: [r"\b(config|setting|parameter|adjust)\b"],
            QueryType.BACKTESTING: [r"\b(backtest|historical|simulate)\b"],
            QueryType.PARAMETER_TUNING: [r"\b(optimize|tune|calibrate)\b"],
            QueryType.GENERAL_QUESTION: [r"\b(what|how|why|explain)\b"],
        }
        self.model_preferences: Dict[QueryType, str] = {
            QueryType.MARKET_NEWS: "perplexity",
            QueryType.DATA_CALCULATION: "openai",
            QueryType.STRATEGY_ANALYSIS: "claude",
            QueryType.RISK_ASSESSMENT: "claude",
            QueryType.SYSTEM_CONFIG: "openai",
            QueryType.GENERAL_QUESTION: "claude",
            QueryType.BACKTESTING: "claude",
            QueryType.PARAMETER_TUNING: "claude",
        }

    def classify_query(self, query: str, context: Optional[Dict] = None) -> QueryClassification:
        scores: Dict[QueryType, float] = {qt: 0.0 for qt in QueryType}
        keyword_hits: Dict[QueryType, List[str]] = {qt: [] for qt in QueryType}
        query_lower = query.lower()

        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    scores[query_type] += len(matches)
                    keyword_hits[query_type].extend(matches)

        if context and context.get("last_query_type"):
            last = context["last_query_type"]
            try:
                enum_val = QueryType[last]
                scores[enum_val] += 0.5
            except KeyError:
                pass

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / max(1.0, sum(scores.values()))
        if scores[best_type] == 0:
            best_type = QueryType.GENERAL_QUESTION
            confidence = 0.3

        preferred_model = self.model_preferences[best_type]
        reasoning = self._reason(best_type, keyword_hits[best_type], confidence)
        return QueryClassification(
            query_type=best_type,
            confidence=confidence,
            preferred_model=preferred_model,
            reasoning=reasoning,
            keywords=keyword_hits[best_type],
        )

    @staticmethod
    def _reason(query_type: QueryType, keywords: List[str], confidence: float) -> str:
        keyword_text = ", ".join(keywords) if keywords else "general language"
        return (
            f"Detected {query_type.value.replace('_', ' ')} intent based on {keyword_text}. "
            f"Confidence: {confidence:.2f}."
        )
