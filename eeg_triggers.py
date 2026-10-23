import pyxid2

PULSE_DURATION_MS = 20   # how long the TTL pulse stays high -- keep short and consistent

class EEGTrigger:

    def __init__(self, dev, pulse_duration_ms=PULSE_DURATION_MS):
        """
        Use the StimTracker device that was already discovered
        by motnew.py.

        We intentionally do NOT call pyxid2.get_xid_devices()
        here because the main script already performed the scan.
        """

        if dev is None:
            raise RuntimeError(
                "EEGTrigger requires a StimTracker device."
            )

        self.dev = dev
        self.dev.set_pulse_duration(pulse_duration_ms)

    def send(self, code):
        """
        Fire a TTL pulse encoding `code` as a binary pattern
        across the StimTracker output lines.

        Code 0 is ignored because an all-zero bitmask produces
        no TTL pulse.
        """

        if code == 0:
            return

        self.dev.activate_line(bitmask=int(code))

    def close(self):
        """Clear all StimTracker output lines."""
        self.dev.clear_all_lines()