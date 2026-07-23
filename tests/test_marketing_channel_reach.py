"""Channel reach helpers for Campaign Execution (all executable channels)."""

from services.marketing_execution import (
    EXECUTABLE_CHANNELS,
    INAPP_CHANNELS,
    OUTBOUND_CHANNELS,
    approval_required_for_channel,
    channel_reach_note,
)


def test_all_executable_channels_have_reach_notes():
    assert EXECUTABLE_CHANNELS == OUTBOUND_CHANNELS | INAPP_CHANNELS
    for ch in sorted(EXECUTABLE_CHANNELS):
        note = channel_reach_note(ch)
        assert note
        assert not note.startswith("Channel `")


def test_approval_required_only_for_outbound():
    for ch in OUTBOUND_CHANNELS:
        assert approval_required_for_channel(ch) is True
    for ch in INAPP_CHANNELS:
        assert approval_required_for_channel(ch) is False


def test_channel_reach_notes_describe_delivery():
    assert "Mailgun" in channel_reach_note("email")
    assert "Discord" in channel_reach_note("discord")
    assert "Email" in channel_reach_note("mixed")
    assert "discount" in channel_reach_note("discount").lower()
    assert "nudge" in channel_reach_note("in_app").lower() or "wallet" in channel_reach_note("in_app").lower()
