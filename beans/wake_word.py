"""
Wake-word detection via Porcupine. SCAFFOLD — not yet implemented.
Will be wired up in v1.1. See README roadmap.
"""


class WakeWordListener:
    """Toggles AppState.visible when 'Hey Beans' is detected."""

    def __init__(self, state, model_path: str, access_key: str):
        self.state = state
        self.model_path = model_path
        self.access_key = access_key

    def start(self):
        raise NotImplementedError("Wake-word support arrives in v1.1")

    def stop(self):
        pass
