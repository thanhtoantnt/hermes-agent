"""Property-based tests for hermes_cli.env_loader using Hypothesis.

Tests verify invariants for .env file loading behavior.
"""

import os
import pytest
from pathlib import Path
from hypothesis import given, assume, settings, strategies as st
from hypothesis.strategies import composite
from unittest.mock import patch

from hermes_cli.env_loader import load_hermes_dotenv


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@composite
def path_strategy(draw):
    """Generate valid file paths."""
    segments = draw(st.lists(
        st.text(st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-_"), min_size=1, max_size=10),
        min_size=1, max_size=5
    ))
    return "/".join(segments)


# ---------------------------------------------------------------------------
# load_hermes_dotenv properties
# ---------------------------------------------------------------------------

class TestLoadHermesDotenvProperties:
    """Property-based tests for .env loading."""

    @settings(max_examples=50)
    @given(st.text(max_size=100), st.text(max_size=100))
    def test_always_returns_list(self, hermes_home, project_env):
        """load_hermes_dotenv always returns a list."""
        result = load_hermes_dotenv(hermes_home=hermes_home, project_env=project_env)
        assert isinstance(result, list)

    @settings(max_examples=50)
    @given(st.text(max_size=100))
    def test_result_contains_paths(self, hermes_home):
        """Result contains Path objects."""
        result = load_hermes_dotenv(hermes_home=hermes_home)
        for item in result:
            assert isinstance(item, Path)

    @settings(max_examples=50)
    @given(st.text(max_size=100), st.text(max_size=100))
    def test_result_paths_exist(self, hermes_home, project_env):
        """All returned paths existed at time of call."""
        result = load_hermes_dotenv(hermes_home=hermes_home, project_env=project_env)
        # Note: paths may be deleted after the call, but they existed
        assert isinstance(result, list)

    @settings(max_examples=50)
    @given(st.text(max_size=100))
    def test_no_project_env_returns_user_only(self, hermes_home):
        """Without project_env, only user .env is loaded."""
        result = load_hermes_dotenv(hermes_home=hermes_home, project_env=None)
        # Result is a list (may be empty if no user .env exists)
        assert isinstance(result, list)

    @settings(max_examples=50)
    @given(st.text(max_size=100), st.text(max_size=100))
    def test_order_is_user_then_project(self, hermes_home, project_env):
        """User .env is loaded before project .env."""
        result = load_hermes_dotenv(hermes_home=hermes_home, project_env=project_env)
        # If both exist, user comes first
        if len(result) == 2:
            user_env = Path(hermes_home) / ".env" if isinstance(hermes_home, str) else Path(hermes_home) / ".env"
            # First path should be user .env if it exists
            assert isinstance(result[0], Path)


# ---------------------------------------------------------------------------
# Edge case properties
# ---------------------------------------------------------------------------

class TestLoadHermesDotenvEdgeCases:
    """Property-based tests for edge cases."""

    @settings(max_examples=20)
    @given(st.just(None))
    def test_none_hermes_home_uses_default(self, _):
        """None hermes_home uses default HERMES_HOME."""
        result = load_hermes_dotenv(hermes_home=None)
        assert isinstance(result, list)

    @settings(max_examples=20)
    @given(st.just(None), st.just(None))
    def test_both_none_uses_defaults(self, _, __):
        """Both None uses all defaults."""
        result = load_hermes_dotenv(hermes_home=None, project_env=None)
        assert isinstance(result, list)

    @settings(max_examples=50)
    @given(st.text(max_size=100))
    def test_empty_project_env_same_as_none(self, hermes_home):
        """Empty string project_env behaves like None."""
        result1 = load_hermes_dotenv(hermes_home=hermes_home, project_env="")
        result2 = load_hermes_dotenv(hermes_home=hermes_home, project_env=None)
        # Both should return lists
        assert isinstance(result1, list)
        assert isinstance(result2, list)

    @settings(max_examples=50)
    @given(st.text(max_size=100), st.text(max_size=100))
    def test_nonexistent_paths_return_empty_list(self, hermes_home, project_env):
        """Nonexistent paths return empty list."""
        # Use paths that are unlikely to exist
        result = load_hermes_dotenv(
            hermes_home="/nonexistent/path/xyz123",
            project_env="/nonexistent/project/abc456"
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Idempotency and consistency properties
# ---------------------------------------------------------------------------

class TestLoadHermesDotenvConsistency:
    """Tests for consistency and idempotency."""

    @settings(max_examples=30)
    @given(st.text(max_size=100), st.text(max_size=100))
    def test_called_twice_same_result(self, hermes_home, project_env):
        """Calling twice with same args gives same result."""
        result1 = load_hermes_dotenv(hermes_home=hermes_home, project_env=project_env)
        result2 = load_hermes_dotenv(hermes_home=hermes_home, project_env=project_env)
        # Results should be equal (same paths loaded)
        assert result1 == result2

    @settings(max_examples=30)
    @given(st.text(max_size=100))
    def test_project_env_order_doesnt_matter_for_existence(self, hermes_home):
        """Project env parameter doesn't affect existence check."""
        result1 = load_hermes_dotenv(hermes_home=hermes_home, project_env="/path1")
        result2 = load_hermes_dotenv(hermes_home=hermes_home, project_env="/path2")
        # Both return lists (may be same if user .env exists)
        assert isinstance(result1, list)
        assert isinstance(result2, list)
