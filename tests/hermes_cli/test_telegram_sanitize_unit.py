"""Unit test to confirm the consecutive underscore behavior in _sanitize_telegram_name."""

import pytest
from hermes_cli.commands import _sanitize_telegram_name


class TestSanitizeTelegramNameConsecutiveUnderscores:
    """Confirm that consecutive underscores are collapsed by _sanitize_telegram_name."""

    def test_consecutive_underscores_collapsed(self):
        """Input with consecutive underscores is NOT idempotent."""
        input_str = "0__0"
        result = _sanitize_telegram_name(input_str)
        # The function collapses consecutive underscores to single underscore
        assert result == "0_0", f"Expected '0_0', got {result!r}"
        # This means the function is NOT idempotent for this input
        assert result != input_str, "Function should change the input"

    def test_single_underscore_unchanged(self):
        """Input with single underscores IS idempotent."""
        input_str = "0_0"
        result = _sanitize_telegram_name(input_str)
        assert result == input_str, f"Expected {input_str!r}, got {result!r}"

    def test_triple_underscore_collapsed_to_single(self):
        """Triple underscores collapse to single underscore."""
        input_str = "a___b"
        result = _sanitize_telegram_name(input_str)
        assert result == "a_b", f"Expected 'a_b', got {result!r}"

    def test_multiple_groups_of_consecutive_underscores(self):
        """Multiple groups of consecutive underscores all collapse."""
        input_str = "a__b___c"
        result = _sanitize_telegram_name(input_str)
        assert result == "a_b_c", f"Expected 'a_b_c', got {result!r}"

    def test_idempotency_after_first_application(self):
        """Applying the function twice gives the same result as once."""
        input_str = "0__0"
        first = _sanitize_telegram_name(input_str)
        second = _sanitize_telegram_name(first)
        assert first == second, "Function should be idempotent after first application"

    def test_valid_telegram_names_are_idempotent(self):
        """Names without consecutive underscores, leading/trailing underscores are idempotent."""
        valid_names = [
            "valid_name",
            "valid_name_123",
            "a",
            "test_command",
            "mybot",
        ]
        for name in valid_names:
            result = _sanitize_telegram_name(name)
            assert result == name, f"Expected {name!r} to be unchanged, got {result!r}"

    def test_invalid_names_are_not_idempotent(self):
        """Names with consecutive underscores or leading/trailing underscores are changed."""
        invalid_names = [
            ("0__0", "0_0"),      # consecutive underscores
            ("_leading", "leading"),  # leading underscore
            ("trailing_", "trailing"),  # trailing underscore
            ("_both_", "both"),   # both leading and trailing
            ("a___b", "a_b"),     # triple underscores
        ]
        for input_str, expected in invalid_names:
            result = _sanitize_telegram_name(input_str)
            assert result == expected, f"Expected {expected!r}, got {result!r}"
            assert result != input_str, f"Input {input_str!r} should be changed"
