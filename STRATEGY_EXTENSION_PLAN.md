## Strategy Extension Plan: Universe-Driven Ideas

We currently generate recommendations only off live positions. To surface ideas even when the account is flat, we need to analyze a curated ticker universe and feed those option chains into the existing strategy engine.

### Goals

1. **Universe scan**: Pull option chains for a configurable list of symbols (default list or watchlist) using the Tastytrade SDK.
2. **Model ensemble hookup**: For each underlying, run our Model Ensemble logic to compute edge, consensus price, and strategy type.
3. **Strategy translation**: Convert the ensemble’s `recommended_strategy` into specific trade structures (verticals, condors, directional spreads, etc.) using our existing `StrategyEngine`.
4. **UI updates**: Present universe-based ideas alongside position-based ones, clearly labeling which proposals come from holdings vs opportunity scans.

### Implementation Steps

1. **Universe configuration**
   - Add a configuration file or section (e.g., `frontend/config/universe.py`) listing default symbols.
   - Allow overrides via Streamlit sidebar or `.env`.

2. **Option chain retrieval**
   - Extend `StrategyEngine` with a method to fetch option chains for each universe symbol.
   - Use the Tastytrade SDK’s `get_option_chain` to fetch nested chains.
   - Cache results per refresh to avoid repeated API hits.

3. **Model ensemble integration**
   - Instantiate the backend’s `ModelEnsemble` class with the desired risk profile.
   - Feed the option chain and intraday market data into `analyze_universe` to get ensemble predictions.
   - Filter predictions based on edge magnitude, confidence, and desired strategy type.

4. **Trade synthesis**
   - Map ensemble `recommended_strategy` enums to concrete trade structures (e.g., `VOLATILITY_ARBITRAGE` → iron condor).
   - Reuse `StrategyEngine` helpers to generate order payloads and metrics (max gain, max loss, breakeven).
   - Ensure every suggestion includes category, rationale, and sample order JSON.

5. **UI integration**
   - Update `TradingBotAPI` to return both position-driven and universe-driven ideas.
   - In `frontend/app.py`, add sections for “Live Positions Ideas” vs “Universe Opportunities”.
   - Provide filters (by strategy type, category, symbol) for easier triage.

6. **Performance considerations**
   - Option chains can be heavy; consider limiting expiries (next 30-45 days) and strikes (±10% OTM).
   - Optionally run the universe scan asynchronously or on a timed refresh to avoid blocking the UI.

### Future Enhancements

1. **Watchlist download** from the Tastytrade account instead of hard-coded lists.
2. **Backtesting hooks** to evaluate ensemble performance per symbol over time.
3. **DXLink streaming** for real-time mark updates and faster edge detection.
4. **Signal persistence** to track which ideas were generated and acted upon.

This plan keeps the existing strategy stack intact while adding the missing exploration layer, so the bot can surface actionable trades even before positions are opened.
