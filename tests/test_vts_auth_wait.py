"""The VTS auth handshake needs a timeout a human can beat.

``VTubeStudioClient`` uses ``timeout=1.0`` for every request. That is fine for
parameter injection (a few milliseconds of work), but ``authenticate()`` sends
``AuthenticationTokenRequest``, which makes VTube Studio **show a popup and wait
for a human click**. Measured on this machine (2026-09-16):

    [13:05:15] Plugin "..." requested an API authentication token. Triggering authentication popup.
    [13:05:16] A plugin has disconnected from VTube Studio API.        <- bridge gave up after ~1s
    [13:05:21] User granted authentication request ... Creating token. <- the click arrived 5s later
    [13:05:21] ... Authenticated plugin "..." Returning token.         <- returned to a closed socket

So the documented first-run flow ("VTS 会弹出授权请求，点允许") could never succeed: the click
always lands after the socket is already gone, the token is never written to the token cache,
and every later run asks again. The fix is to wait long enough around that one request.

These tests drive ``authenticate()`` with a fake websocket that records the timeouts it is
given and answers the way VTube Studio would, so no VTS instance is needed.
"""
import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

_IMPORT_ERROR = None
try:
    from vtube_studio_bridge.vtube_studio_bridge import VTubeStudioClient  # noqa: E402
except Exception as exc:  # pragma: no cover - only on a broken environment
    _IMPORT_ERROR = exc
    VTubeStudioClient = None

MIN_SECONDS_FOR_A_HUMAN_CLICK = 60.0


class FakeWebSocket:
    """Stands in for the websocket, recording timeouts and answering like VTS."""

    def __init__(self):
        self.timeouts = []
        self.sent = []
        self.closed = False

    def settimeout(self, value):
        self.timeouts.append(value)

    def send(self, payload):
        self.sent.append(json.loads(payload))

    def recv(self):
        last = self.sent[-1]["messageType"]
        if last == "AuthenticationTokenRequest":
            return json.dumps({"messageType": "AuthenticationTokenResponse",
                               "data": {"authenticationToken": "test-token"}})
        if last == "AuthenticationRequest":
            return json.dumps({"messageType": "AuthenticationResponse",
                               "data": {"authenticated": True}})
        return json.dumps({"messageType": "UnknownResponse", "data": {}})

    def close(self, timeout=0.2):
        self.closed = True

    def message_types(self):
        return [m["messageType"] for m in self.sent]


def make_client(stored_token=None):
    client = VTubeStudioClient()
    fake = FakeWebSocket()
    client.ws = fake
    client._load_token = lambda: stored_token
    saved = []
    client._save_token = lambda token: saved.append(token)
    return client, fake, saved


@unittest.skipIf(VTubeStudioClient is None,
                 f"bridge module could not be imported: {_IMPORT_ERROR}")
class TestVtsAuthWait(unittest.TestCase):
    def test_token_request_waits_long_enough_for_a_human_to_click_the_popup(self):
        client, fake, _ = make_client()

        client.authenticate()

        self.assertTrue(
            fake.timeouts,
            "authenticate() must raise the socket timeout around the token request, "
            "otherwise the popup click always arrives after the socket is gone",
        )
        self.assertGreaterEqual(
            max(fake.timeouts), MIN_SECONDS_FOR_A_HUMAN_CLICK,
            f"the token request has to wait at least {MIN_SECONDS_FOR_A_HUMAN_CLICK:.0f}s "
            "for the user to click 'Allow' in VTube Studio",
        )

    def test_the_original_short_timeout_is_restored_after_the_token_request(self):
        client, fake, _ = make_client()

        client.authenticate()

        self.assertEqual(fake.timeouts[-1], client.timeout,
                         "normal requests must go back to the short timeout")

    def test_an_existing_token_skips_the_popup_and_does_not_change_the_timeout(self):
        client, fake, saved = make_client(stored_token="already-have-one")

        client.authenticate()

        self.assertEqual(fake.message_types(), ["AuthenticationRequest"],
                         "a cached token must authenticate directly, without asking for a new one")
        self.assertEqual(fake.timeouts, [],
                         "no popup means no reason to touch the socket timeout")

    def test_the_new_token_is_saved_for_later_runs(self):
        client, fake, saved = make_client()

        client.authenticate()

        self.assertEqual(saved, ["test-token"],
                         "the approved token must be cached, or every run asks the user again")
        self.assertEqual(fake.message_types(),
                         ["AuthenticationTokenRequest", "AuthenticationRequest"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
