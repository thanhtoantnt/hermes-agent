"""Property-based tests for hermes_cli.profiles using Hypothesis.

Tests verify invariants for profile name validation, path resolution, and operations.
"""

import pytest
from hypothesis import given, assume, strategies as st
from hypothesis.strategies import composite

from hermes_cli.profiles import (
    validate_profile_name,
    get_profile_dir,
    profile_exists,
    check_alias_collision,
    _PROFILE_ID_RE,
    _RESERVED_NAMES,
    _HERMES_SUBCOMMANDS,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@composite
def valid_profile_name_strategy(draw):
    """Generate valid profile names matching [a-z0-9][a-z0-9_-]{0,63}."""
    first_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    rest_chars = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    first = draw(st.sampled_from(list(first_chars)))
    rest = draw(st.text(st.sampled_from(list(rest_chars)), max_size=63))
    return first + rest


@composite
def invalid_profile_name_strategy(draw):
    """Generate invalid profile names."""
    invalid_types = draw(st.integers(0, 5))
    
    if invalid_types == 0:
        return draw(st.text(st.sampled_from(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")), min_size=1, max_size=10))
    elif invalid_types == 1:
        return draw(st.text(st.just(" "), min_size=1, max_size=10))
    elif invalid_types == 2:
        return "." + draw(st.text(max_size=10))
    elif invalid_types == 3:
        return "-" + draw(st.text(max_size=10))
    elif invalid_types == 4:
        return "_" + draw(st.text(max_size=10))
    else:
        return ""


@composite
def profile_name_strategy(draw):
    """Generate any profile name (valid or invalid)."""
    return draw(st.one_of(
        valid_profile_name_strategy(),
        invalid_profile_name_strategy(),
        st.just("default"),
    ))


# ---------------------------------------------------------------------------
# validate_profile_name properties
# ---------------------------------------------------------------------------

class TestValidateProfileNameProperties:
    """Property-based tests for profile name validation."""

    @given(valid_profile_name_strategy())
    def test_valid_names_accepted(self, name):
        """Valid names pass validation."""
        assume(name != "default")
        validate_profile_name(name)

    @given(invalid_profile_name_strategy())
    def test_invalid_names_rejected(self, name):
        """Invalid names raise ValueError."""
        assume(name != "default")
        try:
            validate_profile_name(name)
            assert False, f"Expected ValueError for {name!r}"
        except ValueError:
            pass

    @given(st.just("default"))
    def test_default_accepted(self, name):
        """'default' is always accepted."""
        validate_profile_name(name)

    @given(st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789")))
    def test_single_char_accepted(self, char):
        """Single alphanumeric character is valid."""
        validate_profile_name(char)

    @given(st.text(st.characters(whitelist_categories=('Ll', 'Nd')), min_size=65, max_size=70))
    def test_too_long_rejected(self, name):
        """Names > 64 chars are rejected."""
        with pytest.raises(ValueError):
            validate_profile_name(name)

    @given(valid_profile_name_strategy())
    def test_valid_matches_regex(self, name):
        """Valid names match the profile ID regex."""
        assume(name != "default")
        assert _PROFILE_ID_RE.match(name)


# ---------------------------------------------------------------------------
# get_profile_dir properties
# ---------------------------------------------------------------------------

class TestGetProfileDirProperties:
    """Property-based tests for profile directory resolution."""

    @given(valid_profile_name_strategy())
    def test_returns_path(self, name):
        """get_profile_dir returns a Path-like object."""
        from pathlib import Path
        result = get_profile_dir(name)
        assert isinstance(result, Path)

    @given(st.just("default"))
    def test_default_returns_hermes_home(self, name):
        """'default' returns the default HERMES_HOME."""
        result = get_profile_dir(name)
        assert result.name == ".hermes" or result.name == "hermes"

    @given(valid_profile_name_strategy())
    def test_named_profile_under_profiles(self, name):
        """Named profiles are under profiles/ directory."""
        assume(name != "default")
        result = get_profile_dir(name)
        parts = result.parts
        assert "profiles" in parts
        assert name in parts


# ---------------------------------------------------------------------------
# profile_exists properties
# ---------------------------------------------------------------------------

class TestProfileExistsProperties:
    """Property-based tests for profile existence check."""

    @given(st.just("default"))
    def test_default_always_exists(self, name):
        """'default' profile always exists."""
        assert profile_exists(name) is True

    @given(valid_profile_name_strategy())
    def test_returns_bool(self, name):
        """profile_exists returns bool."""
        result = profile_exists(name)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# check_alias_collision properties
# ---------------------------------------------------------------------------

class TestAliasCollisionProperties:
    """Property-based tests for alias collision detection."""

    @given(valid_profile_name_strategy())
    def test_returns_string_or_none(self, name):
        """check_alias_collision returns string or None."""
        result = check_alias_collision(name)
        assert result is None or isinstance(result, str)

    @given(st.sampled_from(list(_RESERVED_NAMES)))
    def test_reserved_names_collision(self, name):
        """Reserved names always collide."""
        result = check_alias_collision(name)
        assert result is not None
        assert "reserved" in result.lower()

    @given(st.sampled_from(list(_HERMES_SUBCOMMANDS)))
    def test_subcommands_collide(self, name):
        """Hermes subcommands always collide."""
        result = check_alias_collision(name)
        assert result is not None
        assert "subcommand" in result.lower()


# ---------------------------------------------------------------------------
# Profile name boundary properties
# ---------------------------------------------------------------------------

class TestProfileNameBoundaryProperties:
    """Property-based tests for profile name boundaries."""

    @given(st.just("a" * 64))
    def test_max_length_accepted(self, name):
        """64 chars is the maximum allowed."""
        validate_profile_name(name)

    @given(st.just("a" * 65))
    def test_over_max_rejected(self, name):
        """65 chars is rejected."""
        with pytest.raises(ValueError):
            validate_profile_name(name)

    @given(st.integers(1, 64))
    def test_any_length_up_to_max_accepted(self, length):
        """Any length 1-64 is accepted for valid chars."""
        name = "a" * length
        validate_profile_name(name)


# ---------------------------------------------------------------------------
# Profile name character set properties
# ---------------------------------------------------------------------------

class TestProfileNameCharsetProperties:
    """Property-based tests for profile name character sets."""

    @given(st.text(st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789")), min_size=1, max_size=64))
    def test_alphanumeric_only_accepted(self, name):
        """Pure alphanumeric names are accepted."""
        validate_profile_name(name)

    @given(st.text(st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789-")), min_size=1, max_size=64))
    def test_hyphens_accepted(self, name):
        """Names with hyphens are accepted if they start with alphanumeric."""
        assume(name and name[0].isalnum())
        validate_profile_name(name)

    @given(st.text(st.sampled_from(list("abcdefghijklmnopqrstuvwxyz0123456789_")), min_size=1, max_size=64))
    def test_underscores_accepted(self, name):
        """Names with underscores are accepted if they start with alphanumeric."""
        assume(name and name[0].isalnum())
        validate_profile_name(name)

    @given(st.text(st.sampled_from(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")), min_size=1, max_size=10))
    def test_uppercase_rejected(self, name):
        """Uppercase letters are rejected."""
        with pytest.raises(ValueError):
            validate_profile_name(name)

    @given(st.text(st.just(" "), min_size=1, max_size=10))
    def test_whitespace_rejected(self, name):
        """Whitespace is rejected."""
        with pytest.raises(ValueError):
            validate_profile_name(name)
