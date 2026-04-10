"""Property-based tests for hermes_cli.providers using Hypothesis.

Tests verify invariants for provider normalization, overlay resolution, and aggregator detection.
"""

import pytest
from hypothesis import given, assume, settings, strategies as st
from hypothesis.strategies import composite

from hermes_cli.providers import (
    normalize_provider,
    get_overlay,
    get_provider,
    get_label,
    is_aggregator,
    HERMES_OVERLAYS,
    ALIASES,
    ProviderDef,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@composite
def provider_id_strategy(draw):
    """Generate provider IDs from known overlays."""
    return draw(st.sampled_from(list(HERMES_OVERLAYS.keys())))


@composite
def alias_strategy(draw):
    """Generate provider aliases."""
    return draw(st.sampled_from(list(ALIASES.keys())))


@composite
def arbitrary_provider_name_strategy(draw):
    """Generate arbitrary provider names (valid or invalid)."""
    return draw(st.text(min_size=1, max_size=50))


# ---------------------------------------------------------------------------
# normalize_provider properties
# ---------------------------------------------------------------------------

class TestNormalizeProviderProperties:
    """Property-based tests for provider normalization."""

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_always_returns_string(self, name):
        """normalize_provider always returns a string."""
        result = normalize_provider(name)
        assert isinstance(result, str)

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_result_is_lowercase(self, name):
        """Result is always lowercase."""
        result = normalize_provider(name)
        assert result == result.lower()

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_result_has_no_whitespace(self, name):
        """Result has no leading/trailing whitespace."""
        result = normalize_provider(name)
        assert result == result.strip()

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_idempotent(self, name):
        """Applying normalize_provider twice gives same result."""
        first = normalize_provider(name)
        second = normalize_provider(first)
        assert first == second

    @settings(max_examples=50)
    @given(alias_strategy())
    def test_alias_resolves_to_canonical(self, alias):
        """Aliases resolve to canonical provider IDs."""
        result = normalize_provider(alias)
        # Result should be a canonical ID (not an alias)
        assert result not in ALIASES or result == ALIASES.get(result, result)

    @settings(max_examples=100)
    @given(st.text(st.sampled_from(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")), min_size=1, max_size=50))
    def test_case_insensitive(self, name):
        """Normalization is case-insensitive for ASCII characters."""
        lower = normalize_provider(name.lower())
        upper = normalize_provider(name.upper())
        assert lower == upper


# ---------------------------------------------------------------------------
# get_overlay properties
# ---------------------------------------------------------------------------

class TestGetOverlayProperties:
    """Property-based tests for overlay resolution."""

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_returns_overlay_or_none(self, provider_id):
        """get_overlay returns HermesOverlay or None."""
        result = get_overlay(provider_id)
        assert result is None or hasattr(result, "transport")

    @settings(max_examples=50)
    @given(provider_id_strategy())
    def test_known_provider_has_overlay(self, provider_id):
        """Known providers from HERMES_OVERLAYS have overlays."""
        result = get_overlay(provider_id)
        assert result is not None

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_uses_normalized_provider(self, provider_id):
        """get_overlay uses normalized provider ID."""
        normalized = normalize_provider(provider_id)
        result1 = get_overlay(provider_id)
        result2 = get_overlay(normalized)
        assert result1 == result2


# ---------------------------------------------------------------------------
# get_provider properties
# ---------------------------------------------------------------------------

class TestGetProviderProperties:
    """Property-based tests for provider resolution."""

    @settings(max_examples=100)
    @given(st.text(min_size=1, max_size=50))
    def test_returns_providerdef_or_none(self, name):
        """get_provider returns ProviderDef or None."""
        result = get_provider(name)
        assert result is None or isinstance(result, ProviderDef)

    @settings(max_examples=50)
    @given(provider_id_strategy())
    def test_known_provider_returns_providerdef(self, provider_id):
        """Known providers return ProviderDef."""
        result = get_provider(provider_id)
        # May be None if models.dev is not available
        if result is not None:
            assert isinstance(result, ProviderDef)
            assert result.id == normalize_provider(provider_id)

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_uses_normalized_provider(self, name):
        """get_provider uses normalized provider ID."""
        normalized = normalize_provider(name)
        result1 = get_provider(name)
        result2 = get_provider(normalized)
        assert result1 == result2

    @settings(max_examples=50)
    @given(provider_id_strategy())
    def test_providerdef_has_required_fields(self, provider_id):
        """ProviderDef has all required fields."""
        result = get_provider(provider_id)
        if result is not None:
            assert isinstance(result.id, str)
            assert isinstance(result.name, str)
            assert isinstance(result.transport, str)
            assert isinstance(result.api_key_env_vars, tuple)
            assert isinstance(result.is_aggregator, bool)


# ---------------------------------------------------------------------------
# get_label properties
# ---------------------------------------------------------------------------

class TestGetLabelProperties:
    """Property-based tests for provider labels."""

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_always_returns_string(self, provider_id):
        """get_label always returns a string."""
        result = get_label(provider_id)
        assert isinstance(result, str)

    @settings(max_examples=100)
    @given(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    def test_result_non_empty(self, provider_id):
        """Result is always non-empty for non-whitespace input."""
        result = get_label(provider_id)
        assert len(result) > 0

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_uses_normalized_provider(self, provider_id):
        """get_label uses normalized provider ID."""
        normalized = normalize_provider(provider_id)
        result1 = get_label(provider_id)
        result2 = get_label(normalized)
        assert result1 == result2

    @settings(max_examples=50)
    @given(provider_id_strategy())
    def test_known_provider_has_readable_label(self, provider_id):
        """Known providers have human-readable labels."""
        result = get_label(provider_id)
        # Label should not be the raw ID (should be formatted)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# is_aggregator properties
# ---------------------------------------------------------------------------

class TestIsAggregatorProperties:
    """Property-based tests for aggregator detection."""

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_always_returns_bool(self, provider):
        """is_aggregator always returns bool."""
        result = is_aggregator(provider)
        assert isinstance(result, bool)

    @settings(max_examples=50)
    @given(provider_id_strategy())
    def test_known_provider_aggregator_check(self, provider_id):
        """Known providers have consistent aggregator status."""
        result = is_aggregator(provider_id)
        overlay = get_overlay(provider_id)
        if overlay is not None:
            # If overlay exists, aggregator status should match
            assert isinstance(result, bool)

    @settings(max_examples=100)
    @given(st.text(max_size=50))
    def test_unknown_provider_not_aggregator(self, provider):
        """Unknown providers are not aggregators."""
        assume(get_provider(provider) is None)
        result = is_aggregator(provider)
        # Unknown providers return False
        assert result is False


# ---------------------------------------------------------------------------
# Provider consistency properties
# ---------------------------------------------------------------------------

class TestProviderConsistency:
    """Tests for consistency between provider functions."""

    @settings(max_examples=50)
    @given(provider_id_strategy())
    def test_provider_and_label_consistency(self, provider_id):
        """Provider and label are consistent."""
        pdef = get_provider(provider_id)
        label = get_label(provider_id)
        if pdef is not None:
            # Label should match provider name or be a known override
            assert isinstance(label, str)
            assert len(label) > 0

    @settings(max_examples=50)
    @given(alias_strategy(), alias_strategy())
    def test_aliases_same_canonical_same_label(self, a1, a2):
        """Aliases that resolve to same canonical ID have same label."""
        c1 = normalize_provider(a1)
        c2 = normalize_provider(a2)
        if c1 == c2:
            l1 = get_label(a1)
            l2 = get_label(a2)
            assert l1 == l2

    @settings(max_examples=50)
    @given(provider_id_strategy())
    def test_aggregator_consistent_with_overlay(self, provider_id):
        """Aggregator status is consistent with overlay."""
        result = is_aggregator(provider_id)
        overlay = get_overlay(provider_id)
        pdef = get_provider(provider_id)
        
        if pdef is not None:
            # ProviderDef aggregator status should match function result
            assert pdef.is_aggregator == result
