from enum import Enum
from typing import Dict, Set

# Event types
WHOLE_EVENTS: Set[str] = {
    "Place ping pong ball in cup",
    "Pick up basketball",
    "Put down basketball",
    "Step over cone"
}

REPETITIVE_EVENTS: Set[str] = {
    "Dribbling basketball",
    "Straight walk",
    "Stair up",
    "Stair down"
}

class EventWindowSize(Enum):
    events = {
        "Straight walk": 110,
        "Stair up": 130,
        "Stair down": 110,
        "Pick up basketball": 150,
        "Dribbling basketball": 80,
        "Put down basketball": 180,
        "Place ping pong ball in cup": 250,
        "Step over cone": 160
    }

# Pull sizes directly from enum
EVENT_SIZES: Dict[str, int] = EventWindowSize.events.value