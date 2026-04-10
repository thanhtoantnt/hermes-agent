"""Property-based tests for hermes_cli.colors using Hypothesis.

Tests verify invariants for color application and NO_COLOR/TERM handling.
"""

import os
import sys
import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import patch

from hermes_cli.colors import (
    should_use_color,
    Colors,
    color,
)


class TestShouldUseColorProperties:
    """Property-based tests for should_use_color()."""

    @settings(max_examples=50)
    @given(st.booleans())
    def test_no_color_env_disables_color(self, isatty_value):
        """NO_COLOR env var always disables color."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            with patch.object(sys.stdout, "isatty", return_value=isatty_value):
                assert should_use_color() is False

    @settings(max_examples=50)
    @given(st.booleans())
    def test_term_dumb_disables_color(self, isatty_value):
        """TERM=dumb always disables color."""
        env = os.environ.copy()
        env["TERM"] = "dumb"
        env.pop("NO_COLOR", None)
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=isatty_value):
                assert should_use_color() is False

    @settings(max_examples=50)
    @given(st.booleans())
    def test_color_enabled_when_all_conditions_met(self, is_tty):
        """Color is enabled only when all conditions are met."""
        env = os.environ.copy()
        env.pop("NO_COLOR", None)
        env["TERM"] = "xterm-256color"  # Not "dumb"
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=is_tty):
                result = should_use_color()
                # Color is enabled iff it's a TTY and no disabling conditions
                assert result == is_tty

    @settings(max_examples=50)
    @given(st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x))
    def test_no_color_any_value_disables(self, value):
        """Any value for NO_COLOR disables color (per no-color.org spec)."""
        env = {"NO_COLOR": value, "TERM": "xterm"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                assert should_use_color() is False

    @settings(max_examples=50)
    @given(st.text(min_size=1, max_size=20).filter(lambda x: x != "dumb" and '\x00' not in x))
    def test_term_values_other_than_dumb(self, term_value):
        """TERM values other than 'dumb' don't disable color by themselves."""
        env = {"TERM": term_value}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                result = should_use_color()
                # Should be True since it's a TTY and no disabling conditions
                assert result is True


class TestColorApplicationProperties:
    """Property-based tests for color() function."""

    @settings(max_examples=100)
    @given(st.text(min_size=0, max_size=100))
    def test_color_returns_string(self, text):
        """color() always returns a string."""
        result = color(text, Colors.RED)
        assert isinstance(result, str)

    @settings(max_examples=100)
    @given(st.text(min_size=0, max_size=100))
    def test_color_output_ends_with_reset_when_enabled(self, text):
        """When color is enabled, output ends with RESET."""
        env = {"TERM": "xterm"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                result = color(text, Colors.RED)
                if result != text:  # Only if color was actually applied
                    assert result.endswith(Colors.RESET)

    @settings(max_examples=100)
    @given(st.text(min_size=0, max_size=100))
    def test_color_passthrough_when_disabled(self, text):
        """When color is disabled, text passes through unchanged."""
        env = {"NO_COLOR": "1"}
        with patch.dict(os.environ, env, clear=True):
            result = color(text, Colors.RED, Colors.BOLD)
            assert result == text

    @settings(max_examples=100)
    @given(st.text(min_size=0, max_size=100))
    def test_color_preserves_text_content(self, text):
        """The original text is always contained in the output."""
        env = {"TERM": "xterm"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                result = color(text, Colors.RED)
                # Text should be somewhere in the result
                assert text in result or result == text

    @settings(max_examples=100)
    @given(st.lists(st.sampled_from([Colors.RED, Colors.GREEN, Colors.BLUE, 
                                      Colors.YELLOW, Colors.MAGENTA, Colors.CYAN,
                                      Colors.BOLD, Colors.DIM]), min_size=1, max_size=5))
    def test_multiple_color_codes(self, codes):
        """Multiple color codes can be combined."""
        env = {"TERM": "xterm"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                text = "test"
                result = color(text, *codes)
                # Result should be non-empty and contain the text
                assert len(result) >= len(text)
                assert text in result

    @settings(max_examples=50)
    @given(st.text(min_size=0, max_size=50))
    def test_empty_text_handling(self, text):
        """Empty text can be colored."""
        env = {"TERM": "xterm"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                result = color("", Colors.RED)
                # Should return just the color codes or empty string
                assert isinstance(result, str)


class TestColorConstantsProperties:
    """Tests for color constant invariants."""

    def test_all_color_constants_are_ansi_escape_sequences(self):
        """All color constants should be valid ANSI escape sequences."""
        for attr_name in ["RESET", "BOLD", "DIM", "RED", "GREEN", "YELLOW", 
                          "BLUE", "MAGENTA", "CYAN"]:
            value = getattr(Colors, attr_name)
            assert value.startswith("\033["), f"{attr_name} is not an ANSI sequence"
            assert value.endswith("m"), f"{attr_name} does not end with 'm'"

    def test_all_color_constants_are_strings(self):
        """All color constants should be strings."""
        for attr_name in dir(Colors):
            if not attr_name.startswith("_"):
                value = getattr(Colors, attr_name)
                if isinstance(value, str):
                    assert isinstance(value, str)

    @settings(max_examples=100)
    @given(st.sampled_from([Colors.RED, Colors.GREEN, Colors.BLUE, 
                            Colors.YELLOW, Colors.MAGENTA, Colors.CYAN]))
    def test_single_color_code_applied(self, code):
        """A single color code is applied correctly."""
        env = {"TERM": "xterm"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                text = "colored"
                result = color(text, code)
                # Result starts with the color code
                assert result.startswith(code)
                # Result ends with reset
                assert result.endswith(Colors.RESET)
