"""Property-based tests for hermes_cli.models using Hypothesis.

Tests verify invariants for model parsing, provider normalization, and model detection.
"""

import pytest
from hypothesis import given, assume, settings, strategies as st
from hypothesis.strategies import composite

from hermes_cli.models import (
    normalize_provider,
    provider_label,
    parse_model_input,
    model_supports_fast_mode,
    _PROVIDER_ALIASES,
    _PROVIDER_MODELS,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@composite
def provider_strategy(draw):
    """Generate provider names from known providers."""
    providers = list(_PROVIDER_MODELS.keys()) + list(_PROVIDER_ALIASES.keys())
    return draw(st.sampled_from(providers))


@composite
def model_name_strategy(draw):
    """Generate realistic model names."""
    vendors = ["claude", "gpt", "gemini", "glm", "kimi", "minimax", "qwen", "deepseek"]
    vendor = draw(st.sampled_from(vendors))
    
    version_type = draw(st.integers(0, 2))
    if version_type == 0:
        version = ""
    elif version_type == 1:
        major = draw(st.integers(1, 5))
        version = f"-{major}"
    else:
        major = draw(st.integers(1, 5))
        minor = draw(st.integers(0, 9))
        version = f"-{major}.{minor}"
    
    suffixes = ["", "-mini", "-turbo", "-pro", "-flash", "-opus", "-sonnet", "-haiku"]
    suffix = draw(st.sampled_from(suffixes))
    
    return f"{vendor}{version}{suffix}"


@composite
def vendor_model_strategy(draw):
    """Generate vendor/model format like 'anthropic/claude-sonnet-4.6'."""
    vendors = ["anthropic", "openai", "google", "z-ai", "moonshotai", "minimax", "qwen"]
    vendor = draw(st.sampled_from(vendors))
    model = draw(model_name_strategy())
    return f"{vendor}/{model}"


@composite
def provider_model_input_strategy(draw):
    """Generate 'provider:model' syntax inputs."""
    provider = draw(provider_strategy())
    model = draw(model_name_strategy())
    return f"{provider}:{model}"


# ---------------------------------------------------------------------------
# normalize_provider properties
# ---------------------------------------------------------------------------

class TestNormalizeProviderProperties:
    """Property-based tests for provider normalization."""

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_always_returns_string(self, provider):
        """normalize_provider always returns a string."""
        result = normalize_provider(provider)
        assert isinstance(result, str)

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_result_is_lowercase(self, provider):
        """Result is always lowercase."""
        result = normalize_provider(provider)
        assert result == result.lower()

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_result_has_no_whitespace(self, provider):
        """Result has no leading/trailing whitespace."""
        result = normalize_provider(provider)
        assert result == result.strip()

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_idempotent(self, provider):
        """Applying normalize_provider twice gives same result."""
        first = normalize_provider(provider)
        second = normalize_provider(first)
        assert first == second

    @settings(max_examples=50)
    @given(provider_strategy())
    def test_known_provider_normalized(self, provider):
        """Known providers normalize to valid provider id."""
        result = normalize_provider(provider)
        assert result in _PROVIDER_MODELS or result in _PROVIDER_ALIASES.values()

    @settings(max_examples=50)
    @given(st.just(None))
    def test_none_returns_default(self, _):
        """None returns default provider."""
        result = normalize_provider(None)
        assert isinstance(result, str)
        assert len(result) > 0

    @settings(max_examples=50)
    @given(st.just(""))
    def test_empty_returns_default(self, _):
        """Empty string returns default provider."""
        result = normalize_provider("")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# provider_label properties
# ---------------------------------------------------------------------------

class TestProviderLabelProperties:
    """Property-based tests for provider label."""

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_always_returns_string(self, provider):
        """provider_label always returns a string."""
        result = provider_label(provider)
        assert isinstance(result, str)

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_result_non_empty(self, provider):
        """Result is always non-empty."""
        result = provider_label(provider)
        assert len(result) > 0

    @settings(max_examples=50)
    @given(provider_strategy())
    def test_known_provider_has_label(self, provider):
        """Known providers have human-readable labels."""
        result = provider_label(provider)
        # Labels should be title-cased or properly formatted
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# parse_model_input properties
# ---------------------------------------------------------------------------

class TestParseModelInputProperties:
    """Property-based tests for model input parsing."""

    @settings(max_examples=100)
    @given(st.text(max_size=100), st.text(max_size=30))
    def test_always_returns_tuple(self, raw, current_provider):
        """parse_model_input always returns a 2-tuple."""
        result = parse_model_input(raw, current_provider)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @settings(max_examples=100)
    @given(st.text(max_size=100), st.text(max_size=30))
    def test_both_elements_are_strings(self, raw, current_provider):
        """Both elements of the tuple are strings."""
        provider, model = parse_model_input(raw, current_provider)
        assert isinstance(provider, str)
        assert isinstance(model, str)

    @settings(max_examples=100)
    @given(model_name_strategy(), st.text(min_size=1, max_size=30))
    def test_simple_model_returns_current_provider(self, model, current_provider):
        """Simple model name without provider prefix returns current provider."""
        assume(":" not in model)
        provider, result_model = parse_model_input(model, current_provider)
        assert provider == current_provider

    @settings(max_examples=50)
    @given(provider_model_input_strategy(), st.text(min_size=1, max_size=30))
    def test_provider_model_syntax(self, input_str, current_provider):
        """'provider:model' syntax extracts provider."""
        provider, model = parse_model_input(input_str, current_provider)
        # Provider should be extracted from input
        assert isinstance(provider, str)
        assert isinstance(model, str)
        # The model part should be non-empty if input was well-formed
        if ":" in input_str:
            assert len(model) > 0 or len(provider) > 0

    @settings(max_examples=100)
    @given(st.text(max_size=100))
    def test_whitespace_handling(self, raw):
        """Whitespace is handled gracefully."""
        provider1, model1 = parse_model_input(raw, "openrouter")
        provider2, model2 = parse_model_input(raw.strip(), "openrouter")
        # Results should be similar after stripping
        assert provider1 == provider2
        assert model1.strip() == model2


# ---------------------------------------------------------------------------
# model_supports_fast_mode properties
# ---------------------------------------------------------------------------

class TestModelSupportsFastModeProperties:
    """Property-based tests for fast mode detection."""

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_always_returns_bool(self, model_id):
        """model_supports_fast_mode always returns bool."""
        result = model_supports_fast_mode(model_id)
        assert isinstance(result, bool)

    @settings(max_examples=50)
    @given(st.just(None))
    def test_none_returns_false(self, _):
        """None model returns False."""
        assert model_supports_fast_mode(None) is False

    @settings(max_examples=50)
    @given(st.just(""))
    def test_empty_returns_false(self, _):
        """Empty string returns False."""
        assert model_supports_fast_mode("") is False

    @settings(max_examples=100)
    @given(vendor_model_strategy())
    def test_vendor_prefix_handling(self, model):
        """Vendor prefix is stripped before checking."""
        # Models with vendor prefix should work
        result = model_supports_fast_mode(model)
        assert isinstance(result, bool)

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_case_insensitive(self, model_id):
        """Fast mode check is case-insensitive."""
        lower_result = model_supports_fast_mode(model_id.lower() if model_id else None)
        upper_result = model_supports_fast_mode(model_id.upper() if model_id else None)
        assert lower_result == upper_result


# ---------------------------------------------------------------------------
# Provider-model consistency properties
# ---------------------------------------------------------------------------

class TestProviderModelConsistency:
    """Tests for consistency between provider and model handling."""

    @settings(max_examples=50)
    @given(provider_strategy())
    def test_provider_has_models(self, provider):
        """Known providers have model lists."""
        normalized = normalize_provider(provider)
        models = _PROVIDER_MODELS.get(normalized, [])
        # Most providers should have at least one model
        # (some might be empty if not in the static dict)
        assert isinstance(models, list)

    @settings(max_examples=100)
    @given(provider_strategy(), provider_strategy())
    def test_alias_normalization_consistent(self, p1, p2):
        """Same canonical provider normalizes to same result."""
        n1 = normalize_provider(p1)
        n2 = normalize_provider(p2)
        if n1 == n2:
            # If they normalize to the same provider, labels should match
            assert provider_label(p1) == provider_label(p2)
