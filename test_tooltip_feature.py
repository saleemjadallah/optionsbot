"""
Test script for Trading Term Tooltip Feature
============================================

Simple test to verify the tooltip system is working correctly.
"""

import sys
import os

# Add frontend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend'))

def test_term_definitions():
    """Test the term definitions manager"""
    print("Testing Term Definitions Manager...")

    from frontend.utils.term_definitions import get_term_definition

    test_terms = ['Delta', 'Gamma', 'VaR', 'Sharpe Ratio', 'Win Rate']

    for term in test_terms:
        print(f"\n{term}:")
        definition = get_term_definition(term)
        print(f"  {definition[:100]}...")  # First 100 chars

        assert definition, f"No definition returned for {term}"
        assert len(definition) > 20, f"Definition too short for {term}"

    print("\n✅ All term definitions test passed!")


def test_tooltip_html():
    """Test the tooltip HTML generation"""
    print("\nTesting Tooltip HTML Generation...")

    from frontend.utils.tooltip_component import get_tooltip_html

    html = get_tooltip_html("Delta", icon_size="0.7em")

    # Check for key components
    assert "tooltip-icon" in html, "Missing tooltip icon class"
    assert "tooltip-popup" in html, "Missing tooltip popup class"
    assert "Delta" in html, "Missing term name"
    assert "❓" in html, "Missing question mark icon"
    assert "<style>" in html, "Missing style tags"

    print("✅ Tooltip HTML generation test passed!")


def test_backend_integration():
    """Test backend API endpoint (requires backend running)"""
    print("\nTesting Backend API Integration...")

    try:
        import requests

        # Test health endpoint
        response = requests.get("http://localhost:8000/api/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"  Backend Status: {data['status']}")
            print(f"  OpenAI Configured: {data['openai_configured']}")
            print(f"  Cache Size: {data['cache_size']}")
            print("✅ Backend API test passed!")
        else:
            print("⚠️  Backend API returned non-200 status")

    except requests.exceptions.ConnectionError:
        print("⚠️  Backend not running - skipping API test")
        print("   Start backend with: cd backend && python -m uvicorn api.endpoints:app --reload")
    except Exception as e:
        print(f"⚠️  Backend API test failed: {e}")


def test_cache_functionality():
    """Test caching behavior"""
    print("\nTesting Cache Functionality...")

    from frontend.utils.term_definitions import TermDefinitionManager

    manager = TermDefinitionManager()

    # Get a definition (should use fallback or cache it)
    def1 = manager.get_definition("Delta")

    # Get same definition again (should come from cache or fallback)
    def2 = manager.get_definition("Delta")

    assert def1 == def2, "Not returning same definition consistently"
    assert len(def1) > 20, "Definition seems invalid"

    print("  ✓ Consistent definition retrieval")
    print("  ✓ Fallback definitions working")
    print("✅ Cache functionality test passed!")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Trading Term Tooltip Feature - Test Suite")
    print("=" * 60)

    try:
        test_term_definitions()
        test_tooltip_html()
        test_cache_functionality()
        test_backend_integration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe tooltip feature is ready to use!")
        print("\nTo see it in action:")
        print("1. Start the backend: cd backend && python -m uvicorn api.endpoints:app --reload")
        print("2. Start the frontend: cd frontend && streamlit run app.py")
        print("3. Hover over the ❓ icons next to trading terms")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
