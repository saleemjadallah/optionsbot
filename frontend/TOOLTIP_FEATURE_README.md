# Trading Term Tooltips Feature

## Overview

This feature adds interactive question mark icons (❓) next to trading terminology throughout the frontend dashboard. When users hover over these icons, they see AI-generated definitions explaining what each term means and why it's important.

## Features

- **AI-Powered Definitions**: Uses OpenAI GPT-4o-mini to generate clear, concise trading term definitions
- **Smart Caching**: Definitions are cached both in-memory and on disk to minimize API calls and costs
- **Elegant UI**: Beautiful gradient purple question mark icons that expand to show tooltips on hover
- **Comprehensive Coverage**: Tooltips added to all major trading terms including:
  - Greeks (Delta, Gamma, Theta, Vega, Rho)
  - Risk metrics (VaR, Max Drawdown, Sharpe Ratio, etc.)
  - Performance metrics (Win Rate, Profit Factor, Sortino Ratio, etc.)
  - Trading concepts (Leverage, Implied Volatility, etc.)

## Architecture

### Backend Components

1. **`backend/api/term_definitions_endpoint.py`**
   - FastAPI router that handles definition requests
   - Endpoints:
     - `GET /api/term-definition/{term}` - Get definition for a term
     - `DELETE /api/term-definition/cache` - Clear definition cache
     - `GET /api/health` - Health check

2. **Cache System**
   - In-memory cache for fast repeated lookups
   - Persistent JSON cache stored in `backend/data/term_definitions_cache.json`
   - 30-day cache duration (configurable)

### Frontend Components

1. **`frontend/utils/term_definitions.py`**
   - `TermDefinitionManager` class for managing definitions
   - Fetches from backend API or returns fallback definitions
   - Local caching to reduce backend calls

2. **`frontend/utils/tooltip_component.py`**
   - `render_term_with_tooltip()` - Standalone tooltip component
   - `get_tooltip_html()` - Returns HTML string for inline use
   - `render_inline_term_tooltip()` - Label with adjacent tooltip
   - `render_metric_with_tooltip()` - Streamlit metric with tooltip

### Frontend Integration

The tooltips have been integrated into:
- `dashboard.py` - Main dashboard Greeks display
- `app.py` - Portfolio page, Risk page, Analytics page

## Usage Examples

### Method 1: Inline HTML (Recommended for most cases)

```python
from utils.tooltip_component import get_tooltip_html

# In your Streamlit page
st.markdown(f"""
    **Portfolio Delta** {get_tooltip_html("Delta", icon_size="0.6em")}
""", unsafe_allow_html=True)
```

### Method 2: Standalone Component

```python
from utils.tooltip_component import render_term_with_tooltip

# Render standalone tooltip
render_term_with_tooltip("Gamma", icon_size="0.7em")
```

### Method 3: With Metrics

```python
from utils.tooltip_component import render_metric_with_tooltip

# Render metric with tooltip
render_metric_with_tooltip(
    label="Sharpe Ratio",
    value="1.84",
    term="Sharpe Ratio",
    delta="+0.12"
)
```

## Configuration

### Environment Variables

Ensure your OpenAI API key is set in `backend/.env`:

```bash
OPENAI_API_KEY=sk-...your-key-here
```

### Customization

You can customize tooltip appearance in `tooltip_component.py`:

- `icon` - Change the icon (default: "❓")
- `icon_size` - Adjust icon size (default: "0.7em")
- `tooltip_width` - Adjust tooltip popup width (default: "350px")
- Colors and styles in the CSS sections

### Default Definitions

The system includes fallback definitions for 30+ common trading terms in `term_definitions.py`. These are used when:
- The backend is unavailable
- OpenAI API is not configured
- Network issues prevent API calls

## How It Works

1. **User hovers over a ❓ icon** in the UI
2. **Frontend checks local cache** for the definition
3. **If not cached, requests from backend API** at `/api/term-definition/{term}`
4. **Backend checks its cache** (in-memory and disk)
5. **If not cached, calls OpenAI API** to generate definition
6. **Definition is cached** at both backend and frontend levels
7. **Tooltip displays** with the definition

## Cost Optimization

- **Caching**: Definitions are cached for 30 days, minimizing API calls
- **Efficient Model**: Uses GPT-4o-mini (cost-effective model)
- **Fallback Definitions**: 30+ pre-defined terms never hit the API
- **Single Request**: Each unique term only calls OpenAI once

### Estimated Costs

- GPT-4o-mini: ~$0.00015 per definition (150 tokens avg)
- With 100 unique trading terms: ~$0.015 total
- With caching: Cost amortized over 30 days

## Adding New Terms

### Option 1: Let AI Generate It

Simply use the tooltip function with any term - it will automatically be generated:

```python
st.markdown(f"**Implied Volatility** {get_tooltip_html('Implied Volatility')}", unsafe_allow_html=True)
```

### Option 2: Add to Default Definitions

Edit `frontend/utils/term_definitions.py` and add to the `defaults` dictionary:

```python
defaults = {
    # ... existing terms ...
    'your_new_term': 'Your custom definition here.',
}
```

## Troubleshooting

### Tooltips Not Showing

1. Check that `unsafe_allow_html=True` is set in `st.markdown()`
2. Verify CSS is being loaded (check browser dev tools)
3. Ensure the tooltip HTML is being generated

### API Errors

1. Verify OpenAI API key is set in backend `.env`
2. Check backend logs for errors
3. Test the endpoint directly: `curl http://localhost:8000/api/term-definition/Delta`

### Cache Issues

Clear the cache manually:
```bash
# Backend cache
rm backend/data/term_definitions_cache.json

# Frontend cache
rm frontend/utils/term_definitions_cache.json
```

Or use the API endpoint:
```bash
curl -X DELETE http://localhost:8000/api/term-definition/cache
```

## Future Enhancements

Potential improvements:
- [ ] Add ability to customize definitions via admin UI
- [ ] Support for multiple languages
- [ ] User feedback on definition quality
- [ ] Analytics on most-viewed terms
- [ ] Integration with external financial glossaries
- [ ] Mobile-optimized tooltips
- [ ] Keyboard navigation support

## Examples in Production

### Greeks Display (dashboard.py:353-411)
Portfolio Delta, Gamma, Theta, and Vega metrics all have tooltips.

### Risk Metrics (app.py:733-752)
VaR, Max Drawdown, and Leverage explanations on the Risk Management page.

### Performance Analytics (app.py:925-956)
Sharpe Ratio, Sortino Ratio, Calmar Ratio, Win Rate, and Profit Factor tooltips.

## License

This feature is part of the OptionsTrader application.
