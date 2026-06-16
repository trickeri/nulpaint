from nulpaint.voice.intent import parse_intent


def test_known_phrase_maps_to_command():
    intent = parse_intent("new layer")
    assert intent is not None
    assert intent.cmd == "layer.add"
    assert intent.args == {"type": "paint"}


def test_phrase_is_case_and_space_insensitive():
    assert parse_intent("  UNDO ").cmd == "edit.undo"


def test_unknown_phrase_returns_none():
    assert parse_intent("make it look cooler") is None
