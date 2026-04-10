"""Property-based tests for hermes_cli.commands using Hypothesis.

These tests verify invariants and properties that should hold for all inputs,
complementing the example-based tests in test_commands.py.
"""

import re
from hypothesis import given, assume, strategies as st
from hypothesis.strategies import composite

from hermes_cli.commands import (
    COMMAND_REGISTRY,
    GATEWAY_KNOWN_COMMANDS,
    _sanitize_telegram_name,
    _clamp_command_names,
    _clamp_telegram_names,
    resolve_command,
    _CMD_NAME_LIMIT,
    _TG_INVALID_CHARS,
    _TG_MULTI_UNDERSCORE,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@composite
def command_name_strategy(draw):
    """Generate valid command names: lowercase a-z, 0-9, hyphens, underscores."""
    alphabet = st.characters(
        whitelist_categories=('Ll', 'Nd'),
        whitelist_characters=('-', '_'),
    )
    first = draw(st.characters(whitelist_categories=('Ll', 'Nd')))
    rest = draw(st.text(alphabet, max_size=31))
    return first + rest


@composite
def telegram_name_strategy(draw):
    """Generate valid Telegram command names: lowercase a-z, 0-9, underscores."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789_"
    return draw(st.text(st.sampled_from(list(chars)), min_size=1, max_size=32))


@composite
def arbitrary_string_strategy(draw):
    """Generate arbitrary strings including Unicode and special chars."""
    return draw(st.text(max_size=100))


@composite
def name_desc_pairs_strategy(draw):
    """Generate lists of (name, description) pairs for clamping tests."""
    name = draw(st.text(
        st.characters(whitelist_categories=('Ll', 'Nd'), whitelist_characters=('-', '_')),
        min_size=1, max_size=50
    ))
    desc = draw(st.text(max_size=100))
    return (name, desc)


# ---------------------------------------------------------------------------
# _sanitize_telegram_name properties
# ---------------------------------------------------------------------------

class TestSanitizeTelegramNameProperties:
    """Property-based tests for Telegram name sanitization."""

    @given(arbitrary_string_strategy())
    def test_output_only_valid_telegram_chars(self, s):
        """Output must only contain [a-z0-9_]."""
        result = _sanitize_telegram_name(s)
        valid_pattern = re.compile(r"^[a-z0-9_]*$")
        assert valid_pattern.match(result), f"Invalid chars in: {result!r}"

    @given(arbitrary_string_strategy())
    def test_output_never_exceeds_input_length(self, s):
        """Sanitization never increases string length."""
        result = _sanitize_telegram_name(s)
        assert len(result) <= len(s)

    @given(arbitrary_string_strategy())
    def test_no_consecutive_underscores(self, s):
        """Output never has consecutive underscores."""
        result = _sanitize_telegram_name(s)
        assert "__" not in result

    @given(arbitrary_string_strategy())
    def test_no_leading_trailing_underscores(self, s):
        """Output never has leading or trailing underscores."""
        result = _sanitize_telegram_name(s)
        if result:
            assert not result.startswith("_")
            assert not result.endswith("_")

    @given(telegram_name_strategy())
    def test_idempotent_for_valid_names(self, s):
        """Valid Telegram names pass through unchanged."""
        result = _sanitize_telegram_name(s)
        assert result == s.strip("_")

    @given(arbitrary_string_strategy())
    def test_idempotent_after_first_application(self, s):
        """Applying sanitization twice gives same result as once."""
        first = _sanitize_telegram_name(s)
        second = _sanitize_telegram_name(first)
        assert first == second

    @given(st.text(st.characters(whitelist_categories=('Lu',)), min_size=1, max_size=20))
    def test_uppercase_converted_to_lowercase(self, s):
        """All uppercase input becomes lowercase."""
        result = _sanitize_telegram_name(s)
        assert result == result.lower()

    @given(arbitrary_string_strategy())
    def test_hyphens_become_underscores(self, s):
        """Hyphens are converted to underscores."""
        result = _sanitize_telegram_name(s)
        assert "-" not in result

    @given(arbitrary_string_strategy())
    def test_output_is_lowercase(self, s):
        """Output is always lowercase."""
        result = _sanitize_telegram_name(s)
        assert result == result.lower()


# ---------------------------------------------------------------------------
# _clamp_command_names properties
# ---------------------------------------------------------------------------

class TestClampCommandNamesProperties:
    """Property-based tests for command name clamping."""

    @given(st.lists(name_desc_pairs_strategy(), max_size=20))
    def test_all_output_names_within_limit(self, pairs):
        """All output names are <= 32 characters."""
        result = _clamp_command_names(pairs, reserved=set())
        for name, _ in result:
            assert len(name) <= _CMD_NAME_LIMIT

    @given(st.lists(name_desc_pairs_strategy(), max_size=20))
    def test_output_names_are_unique(self, pairs):
        """Output names are unique (no duplicates)."""
        result = _clamp_command_names(pairs, reserved=set())
        names = [name for name, _ in result]
        assert len(names) == len(set(names))

    @given(st.lists(name_desc_pairs_strategy(), max_size=20), st.sets(st.text(max_size=32), max_size=10))
    def test_output_does_not_overlap_reserved(self, pairs, reserved):
        """Output names don't overlap with reserved names."""
        result = _clamp_command_names(pairs, reserved=reserved)
        for name, _ in result:
            assert name not in reserved

    @given(st.lists(name_desc_pairs_strategy(), max_size=20))
    def test_output_size_at_most_input_size(self, pairs):
        """Clamping never adds entries."""
        result = _clamp_command_names(pairs, reserved=set())
        assert len(result) <= len(pairs)

    @given(name_desc_pairs_strategy())
    def test_short_names_unchanged(self, pair):
        """Names <= 32 chars that don't collide are unchanged."""
        name, desc = pair
        assume(len(name) <= _CMD_NAME_LIMIT)
        result = _clamp_command_names([pair], reserved=set())
        if result:
            assert result[0][0] == name

    @given(st.text(min_size=33, max_size=50))
    def test_long_names_truncated(self, long_name):
        """Names > 32 chars are truncated to 32."""
        result = _clamp_command_names([(long_name, "desc")], reserved=set())
        if result:
            assert len(result[0][0]) == _CMD_NAME_LIMIT
            assert result[0][0] == long_name[:_CMD_NAME_LIMIT]


# ---------------------------------------------------------------------------
# resolve_command properties
# ---------------------------------------------------------------------------

class TestResolveCommandProperties:
    """Property-based tests for command resolution."""

    @given(st.text(max_size=50))
    def test_returns_commanddef_or_none(self, s):
        """resolve_command always returns CommandDef or None."""
        result = resolve_command(s)
        assert result is None or hasattr(result, 'name')

    @given(st.text(max_size=50))
    def test_leading_slash_same_as_without(self, s):
        """Leading slash is optional."""
        without = resolve_command(s)
        with_slash = resolve_command("/" + s)
        assert without == with_slash

    @given(st.text(max_size=50))
    def test_case_insensitive(self, s):
        """Resolution is case-insensitive."""
        lower = resolve_command(s.lower())
        upper = resolve_command(s.upper())
        mixed = resolve_command(s)
        assert lower == upper == mixed

    @given(st.just(""))
    def test_empty_returns_none(self, s):
        """Empty string returns None."""
        assert resolve_command(s) is None


# ---------------------------------------------------------------------------
# GATEWAY_KNOWN_COMMANDS properties
# ---------------------------------------------------------------------------

class TestGatewayKnownCommandsProperties:
    """Property-based tests for gateway command set."""

    @given(st.just(None))
    def test_is_frozenset(self, _):
        """GATEWAY_KNOWN_COMMANDS is a frozenset."""
        assert isinstance(GATEWAY_KNOWN_COMMANDS, frozenset)

    @given(st.just(None))
    def test_all_names_are_strings(self, _):
        """All entries are strings."""
        for name in GATEWAY_KNOWN_COMMANDS:
            assert isinstance(name, str)

    @given(st.just(None))
    def test_no_empty_strings(self, _):
        """No empty strings in the set."""
        assert "" not in GATEWAY_KNOWN_COMMANDS

    @given(st.just(None))
    def test_all_lowercase(self, _):
        """All names are lowercase."""
        for name in GATEWAY_KNOWN_COMMANDS:
            assert name == name.lower()


# ---------------------------------------------------------------------------
# Alias resolution properties
# ---------------------------------------------------------------------------

class TestAliasResolutionProperties:
    """Property-based tests for alias resolution."""

    @given(st.sampled_from([cmd for cmd in COMMAND_REGISTRY if cmd.aliases and cmd.name not in ("queue", "quit")]))
    def test_alias_resolves_to_canonical(self, cmd):
        """Every alias resolves to its canonical command."""
        for alias in cmd.aliases:
            result = resolve_command(alias)
            assert result is not None
            assert result.name == cmd.name

    @given(st.sampled_from(COMMAND_REGISTRY))
    def test_canonical_resolves_to_self(self, cmd):
        """Canonical name resolves to itself."""
        result = resolve_command(cmd.name)
        assert result is not None
        assert result.name == cmd.name
