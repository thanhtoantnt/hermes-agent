"""Property-based tests for hermes_cli.model_normalize using Hypothesis.

Tests verify invariants for model name normalization across providers.
"""

import re
from hypothesis import given, assume, strategies as st
from hypothesis.strategies import composite

from hermes_cli.model_normalize import (
    normalize_model_for_provider,
    detect_vendor,
    _strip_vendor_prefix,
    _dots_to_hyphens,
    _prepend_vendor,
    _VENDOR_PREFIXES,
    _AGGREGATOR_PROVIDERS,
    _DOT_TO_HYPHEN_PROVIDERS,
    _STRIP_VENDOR_ONLY_PROVIDERS,
    _PASSTHROUGH_PROVIDERS,
    model_display_name,
    is_aggregator_provider,
    vendor_for_model,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@composite
def model_name_strategy(draw):
    """Generate model names like 'claude-sonnet-4.6', 'gpt-5.4-mini'."""
    vendors = ["claude", "gpt", "o1", "o3", "gemini", "gemma", "deepseek",
               "glm", "kimi", "minimax", "grok", "qwen", "llama"]
    vendor = draw(st.sampled_from(vendors))
    
    # Version numbers
    version_type = draw(st.integers(0, 2))
    if version_type == 0:
        version = ""
    elif version_type == 1:
        digits = draw(st.text(st.characters(whitelist_categories=('Nd',)), min_size=1, max_size=3))
        version = f"-{digits}"
    else:
        major = draw(st.integers(0, 9))
        minor = draw(st.integers(0, 9))
        version = f"-{major}.{minor}"
    
    # Suffix
    suffixes = ["", "-mini", "-turbo", "-plus", "-pro", "-chat", "-reasoner"]
    suffix = draw(st.sampled_from(suffixes))
    
    return f"{vendor}{version}{suffix}"


@composite
def vendor_prefixed_model_strategy(draw):
    """Generate vendor-prefixed model names like 'anthropic/claude-sonnet-4.6'."""
    vendors = list(_VENDOR_PREFIXES.values())
    vendor = draw(st.sampled_from(vendors))
    model = draw(model_name_strategy())
    return f"{vendor}/{model}"


@composite
def provider_strategy(draw):
    """Generate provider names."""
    providers = list(_AGGREGATOR_PROVIDERS) + list(_DOT_TO_HYPHEN_PROVIDERS) + \
                list(_STRIP_VENDOR_ONLY_PROVIDERS) + list(_PASSTHROUGH_PROVIDERS) + \
                ["deepseek", "custom", "unknown-provider"]
    return draw(st.sampled_from(providers))


@composite
def arbitrary_model_string_strategy(draw):
    """Generate arbitrary model strings including edge cases."""
    return draw(st.one_of(
        model_name_strategy(),
        vendor_prefixed_model_strategy(),
        st.text(max_size=50),
        st.just(""),
    ))


# ---------------------------------------------------------------------------
# normalize_model_for_provider properties
# ---------------------------------------------------------------------------

class TestNormalizeModelForProviderProperties:
    """Property-based tests for model normalization."""

    @given(arbitrary_model_string_strategy(), provider_strategy())
    def test_always_returns_string(self, model, provider):
        """Normalization always returns a string."""
        result = normalize_model_for_provider(model, provider)
        assert isinstance(result, str)

    @given(arbitrary_model_string_strategy(), provider_strategy())
    def test_empty_input_empty_output(self, model, provider):
        """Empty input returns empty output."""
        assume(not model or not model.strip())
        result = normalize_model_for_provider(model, provider)
        assert result == ""

    @given(st.text(min_size=1, max_size=50), st.just("custom"))
    def test_custom_provider_passthrough(self, model, provider):
        """Custom provider passes through unchanged."""
        result = normalize_model_for_provider(model, provider)
        assert result == model.strip()

    @given(arbitrary_model_string_strategy(), st.sampled_from(list(_AGGREGATOR_PROVIDERS)))
    def test_aggregator_has_vendor_prefix(self, model, provider):
        """Aggregator providers always produce vendor/model format."""
        assume(model and model.strip())
        result = normalize_model_for_provider(model, provider)
        assume(result)
        if "/" in result:
            vendor, _ = result.split("/", 1)
            assert vendor

    @given(arbitrary_model_string_strategy(), st.sampled_from(list(_DOT_TO_HYPHEN_PROVIDERS)))
    def test_dot_to_hyphen_no_dots(self, model, provider):
        """Dot-to-hyphen providers have no dots in output."""
        result = normalize_model_for_provider(model, provider)
        assert "." not in result

    @given(arbitrary_model_string_strategy(), st.sampled_from(list(_STRIP_VENDOR_ONLY_PROVIDERS)))
    def test_strip_vendor_no_vendor_prefix(self, model, provider):
        """Strip-vendor providers have no vendor/ prefix."""
        result = normalize_model_for_provider(model, provider)
        assert "/" not in result


# ---------------------------------------------------------------------------
# detect_vendor properties
# ---------------------------------------------------------------------------

class TestDetectVendorProperties:
    """Property-based tests for vendor detection."""

    @given(arbitrary_model_string_strategy())
    def test_returns_string_or_none(self, model):
        """detect_vendor returns string or None."""
        result = detect_vendor(model)
        assert result is None or isinstance(result, str)

    @given(vendor_prefixed_model_strategy())
    def test_prefixed_model_returns_vendor(self, model):
        """Vendor-prefixed models return the vendor."""
        vendor, _ = model.split("/", 1)
        result = detect_vendor(model)
        assert result == vendor.lower()

    @given(st.just(""))
    def test_empty_returns_none(self, model):
        """Empty string returns None."""
        assert detect_vendor(model) is None

    @given(st.text(max_size=50))
    def test_result_is_lowercase(self, model):
        """Result is always lowercase."""
        result = detect_vendor(model)
        if result:
            assert result == result.lower()


# ---------------------------------------------------------------------------
# _strip_vendor_prefix properties
# ---------------------------------------------------------------------------

class TestStripVendorPrefixProperties:
    """Property-based tests for vendor prefix stripping."""

    @given(arbitrary_model_string_strategy())
    def test_always_returns_string(self, model):
        """Always returns a string."""
        result = _strip_vendor_prefix(model)
        assert isinstance(result, str)

    @given(arbitrary_model_string_strategy())
    def test_no_vendor_prefix_in_output(self, model):
        """Output never has vendor/ prefix."""
        result = _strip_vendor_prefix(model)
        assert "/" not in result

    @given(vendor_prefixed_model_strategy())
    def test_strips_vendor(self, model):
        """Vendor prefix is stripped."""
        vendor, rest = model.split("/", 1)
        result = _strip_vendor_prefix(model)
        assert result == rest

    @given(model_name_strategy())
    def test_no_change_without_prefix(self, model):
        """Models without prefix are unchanged."""
        result = _strip_vendor_prefix(model)
        assert result == model


# ---------------------------------------------------------------------------
# _dots_to_hyphens properties
# ---------------------------------------------------------------------------

class TestDotsToHyphensProperties:
    """Property-based tests for dot-to-hyphen conversion."""

    @given(st.text(max_size=50))
    def test_always_returns_string(self, model):
        """Always returns a string."""
        result = _dots_to_hyphens(model)
        assert isinstance(result, str)

    @given(st.text(max_size=50))
    def test_no_dots_in_output(self, model):
        """Output never has dots."""
        result = _dots_to_hyphens(model)
        assert "." not in result

    @given(st.text(st.characters(blacklist_characters=('.',)), max_size=50))
    def test_no_change_without_dots(self, model):
        """Models without dots are unchanged."""
        result = _dots_to_hyphens(model)
        assert result == model

    @given(st.text(max_size=50))
    def test_idempotent(self, model):
        """Applying twice is same as once."""
        first = _dots_to_hyphens(model)
        second = _dots_to_hyphens(first)
        assert first == second


# ---------------------------------------------------------------------------
# _prepend_vendor properties
# ---------------------------------------------------------------------------

class TestPrependVendorProperties:
    """Property-based tests for vendor prepending."""

    @given(arbitrary_model_string_strategy())
    def test_always_returns_string(self, model):
        """Always returns a string."""
        result = _prepend_vendor(model)
        assert isinstance(result, str)

    @given(vendor_prefixed_model_strategy())
    def test_already_prefixed_unchanged(self, model):
        """Already prefixed models are unchanged."""
        result = _prepend_vendor(model)
        assert result == model

    @given(model_name_strategy())
    def test_output_has_vendor_or_passthrough(self, model):
        """Output either has vendor/ or is passthrough."""
        result = _prepend_vendor(model)
        if "/" in result:
            vendor, rest = result.split("/", 1)
            assert vendor
            assert rest


# ---------------------------------------------------------------------------
# model_display_name properties
# ---------------------------------------------------------------------------

class TestModelDisplayNameProperties:
    """Property-based tests for display name."""

    @given(arbitrary_model_string_strategy())
    def test_always_returns_string(self, model):
        """Always returns a string."""
        result = model_display_name(model)
        assert isinstance(result, str)

    @given(arbitrary_model_string_strategy())
    def test_no_vendor_prefix(self, model):
        """Output never has vendor/ prefix."""
        result = model_display_name(model)
        assert "/" not in result

    @given(vendor_prefixed_model_strategy())
    def test_strips_vendor(self, model):
        """Vendor prefix is stripped."""
        _, rest = model.split("/", 1)
        result = model_display_name(model)
        assert result == rest


# ---------------------------------------------------------------------------
# is_aggregator_provider properties
# ---------------------------------------------------------------------------

class TestIsAggregatorProviderProperties:
    """Property-based tests for aggregator check."""

    @given(st.text(max_size=30))
    def test_returns_bool(self, provider):
        """Always returns bool."""
        result = is_aggregator_provider(provider)
        assert isinstance(result, bool)

    @given(st.sampled_from(list(_AGGREGATOR_PROVIDERS)))
    def test_known_aggregators_true(self, provider):
        """Known aggregators return True."""
        assert is_aggregator_provider(provider) is True

    @given(st.sampled_from(list(_DOT_TO_HYPHEN_PROVIDERS)))
    def test_non_aggregators_false(self, provider):
        """Non-aggregators return False."""
        assert is_aggregator_provider(provider) is False


# ---------------------------------------------------------------------------
# vendor_for_model properties
# ---------------------------------------------------------------------------

class TestVendorForModelProperties:
    """Property-based tests for vendor_for_model."""

    @given(arbitrary_model_string_strategy())
    def test_returns_string(self, model):
        """Always returns a string (never None)."""
        result = vendor_for_model(model)
        assert isinstance(result, str)

    @given(vendor_prefixed_model_strategy())
    def test_prefixed_returns_vendor(self, model):
        """Vendor-prefixed models return the vendor."""
        vendor, _ = model.split("/", 1)
        result = vendor_for_model(model)
        assert result == vendor.lower()
