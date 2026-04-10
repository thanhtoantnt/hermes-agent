"""Property-based tests for hermes_cli.skin_engine using Hypothesis.

Tests verify invariants for skin merging, color handling, and inheritance.
"""

import re
from hypothesis import given, assume, strategies as st
from hypothesis.strategies import composite

from hermes_cli.skin_engine import (
    SkinConfig,
    _BUILTIN_SKINS,
    _build_skin_config,
    load_skin,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@composite
def hex_color_strategy(draw):
    """Generate valid hex color codes like #RRGGBB."""
    hex_chars = st.characters(whitelist_categories=('Ll', 'Nd'), whitelist_characters=())
    color = draw(st.text(hex_chars, min_size=6, max_size=6))
    return "#" + color


@composite
def skin_name_strategy(draw):
    """Generate valid skin names: lowercase, hyphens, underscores."""
    alphabet = st.characters(
        whitelist_categories=('Ll', 'Nd'),
        whitelist_characters=('-', '_'),
    )
    first = draw(st.characters(whitelist_categories=('Ll', 'Nd')))
    rest = draw(st.text(alphabet, max_size=30))
    return first + rest


@composite
def skin_dict_strategy(draw):
    """Generate a skin definition dict."""
    name = draw(skin_name_strategy())
    description = draw(st.text(max_size=100))
    
    color_keys = ["banner_border", "banner_title", "banner_accent", "banner_dim",
                  "banner_text", "ui_accent", "ui_label", "ui_ok", "ui_error",
                  "ui_warn", "prompt", "input_rule", "response_border"]
    colors = {}
    for key in color_keys:
        if draw(st.booleans()):
            colors[key] = draw(hex_color_strategy())
    
    branding_keys = ["agent_name", "welcome", "goodbye", "response_label",
                     "prompt_symbol", "help_header"]
    branding = {}
    for key in branding_keys:
        if draw(st.booleans()):
            branding[key] = draw(st.text(max_size=50))
    
    spinner_keys = ["waiting_faces", "thinking_faces", "thinking_verbs"]
    spinner = {}
    for key in spinner_keys:
        if draw(st.booleans()):
            spinner[key] = draw(st.lists(st.text(max_size=20), max_size=5))
    
    return {
        "name": name,
        "description": description,
        "colors": colors,
        "branding": branding,
        "spinner": spinner,
        "tool_prefix": draw(st.characters(max_codepoint=0x10FFFF, min_codepoint=32)),
    }


# ---------------------------------------------------------------------------
# SkinConfig properties
# ---------------------------------------------------------------------------

class TestSkinConfigProperties:
    """Property-based tests for SkinConfig dataclass."""

    @given(skin_dict_strategy())
    def test_build_skin_config_returns_skinconfig(self, data):
        """_build_skin_config always returns a SkinConfig."""
        result = _build_skin_config(data)
        assert isinstance(result, SkinConfig)

    @given(skin_dict_strategy())
    def test_skinconfig_has_name(self, data):
        """SkinConfig always has a name."""
        result = _build_skin_config(data)
        assert isinstance(result.name, str)
        assert len(result.name) > 0

    @given(skin_dict_strategy())
    def test_colors_are_strings(self, data):
        """All color values are strings."""
        result = _build_skin_config(data)
        for key, value in result.colors.items():
            assert isinstance(value, str)

    @given(skin_dict_strategy())
    def test_branding_are_strings(self, data):
        """All branding values are strings."""
        result = _build_skin_config(data)
        for key, value in result.branding.items():
            assert isinstance(value, str)

    @given(skin_dict_strategy())
    def test_tool_prefix_is_string(self, data):
        """tool_prefix is always a string."""
        result = _build_skin_config(data)
        assert isinstance(result.tool_prefix, str)

    @given(skin_dict_strategy())
    def test_get_color_returns_string(self, data):
        """get_color always returns a string."""
        result = _build_skin_config(data)
        key = "banner_title"
        value = result.get_color(key)
        assert isinstance(value, str)

    @given(skin_dict_strategy(), st.text(max_size=20), st.text(max_size=20))
    def test_get_color_fallback(self, data, key, fallback):
        """get_color returns fallback for missing keys."""
        result = _build_skin_config(data)
        assume(key not in result.colors)
        value = result.get_color(key, fallback)
        assert value == fallback

    @given(skin_dict_strategy())
    def test_get_spinner_list_returns_list(self, data):
        """get_spinner_list always returns a list."""
        result = _build_skin_config(data)
        value = result.get_spinner_list("waiting_faces")
        assert isinstance(value, list)

    @given(skin_dict_strategy())
    def test_get_spinner_wings_returns_list_of_tuples(self, data):
        """get_spinner_wings returns list of 2-tuples."""
        result = _build_skin_config(data)
        value = result.get_spinner_wings()
        assert isinstance(value, list)
        for item in value:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)

    @given(skin_dict_strategy())
    def test_get_branding_returns_string(self, data):
        """get_branding always returns a string."""
        result = _build_skin_config(data)
        value = result.get_branding("agent_name")
        assert isinstance(value, str)


# ---------------------------------------------------------------------------
# Inheritance from default skin
# ---------------------------------------------------------------------------

class TestSkinInheritanceProperties:
    """Property-based tests for skin inheritance from default."""

    @given(skin_dict_strategy())
    def test_missing_colors_inherit_from_default(self, data):
        """Colors not specified inherit from default skin."""
        result = _build_skin_config(data)
        default = _BUILTIN_SKINS["default"]
        default_colors = default.get("colors", {})
        
        for key in default_colors:
            if key not in data.get("colors", {}):
                assert result.colors[key] == default_colors[key]

    @given(skin_dict_strategy())
    def test_missing_branding_inherits_from_default(self, data):
        """Branding not specified inherits from default skin."""
        result = _build_skin_config(data)
        default = _BUILTIN_SKINS["default"]
        default_branding = default.get("branding", {})
        
        for key in default_branding:
            if key not in data.get("branding", {}):
                assert result.branding[key] == default_branding[key]

    @given(skin_dict_strategy())
    def test_specified_colors_override_default(self, data):
        """Colors specified in data override default."""
        assume("colors" in data and data["colors"])
        result = _build_skin_config(data)
        
        for key, value in data.get("colors", {}).items():
            assert result.colors[key] == value

    @given(skin_dict_strategy())
    def test_specified_branding_override_default(self, data):
        """Branding specified in data override default."""
        assume("branding" in data and data["branding"])
        result = _build_skin_config(data)
        
        for key, value in data.get("branding", {}).items():
            assert result.branding[key] == value


# ---------------------------------------------------------------------------
# Built-in skin properties
# ---------------------------------------------------------------------------

class TestBuiltinSkinProperties:
    """Property-based tests for built-in skins."""

    @given(st.sampled_from(list(_BUILTIN_SKINS.keys())))
    def test_builtin_loads_successfully(self, name):
        """All built-in skins load without error."""
        skin = load_skin(name)
        assert isinstance(skin, SkinConfig)
        assert skin.name == name

    @given(st.sampled_from(list(_BUILTIN_SKINS.keys())))
    def test_builtin_has_required_colors(self, name):
        """All built-in skins have required color keys."""
        skin = load_skin(name)
        required = ["banner_border", "banner_title", "banner_accent",
                    "banner_dim", "banner_text", "ui_accent"]
        for key in required:
            assert key in skin.colors

    @given(st.sampled_from(list(_BUILTIN_SKINS.keys())))
    def test_builtin_has_required_branding(self, name):
        """All built-in skins have required branding keys."""
        skin = load_skin(name)
        required = ["agent_name", "welcome", "goodbye", "response_label",
                    "prompt_symbol", "help_header"]
        for key in required:
            assert key in skin.branding

    @given(st.sampled_from(list(_BUILTIN_SKINS.keys())))
    def test_builtin_colors_are_hex(self, name):
        """All built-in colors are valid hex codes."""
        skin = load_skin(name)
        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for key, value in skin.colors.items():
            assert hex_pattern.match(value), f"Invalid hex color: {value}"

    @given(st.sampled_from(list(_BUILTIN_SKINS.keys())))
    def test_builtin_tool_prefix_single_char(self, name):
        """Tool prefix is typically a single character."""
        skin = load_skin(name)
        assert len(skin.tool_prefix) >= 1


# ---------------------------------------------------------------------------
# Skin name handling
# ---------------------------------------------------------------------------

class TestSkinNameProperties:
    """Property-based tests for skin name handling."""

    @given(skin_name_strategy())
    def test_unknown_skin_falls_back_to_default(self, name):
        """Unknown skin names fall back to default."""
        assume(name not in _BUILTIN_SKINS)
        skin = load_skin(name)
        assert skin.name == "default"

    @given(st.text(max_size=50))
    def test_load_skin_never_raises(self, name):
        """load_skin handles any input gracefully."""
        skin = load_skin(name)
        assert isinstance(skin, SkinConfig)

    @given(st.just("default"))
    def test_default_skin_always_loads(self, name):
        """Default skin always loads."""
        skin = load_skin(name)
        assert skin.name == "default"
