from pathlib import Path
import yaml

NEGATIVE_INF = -100000.0

PERSPECTIVE_API_KEY = 'AIzaSyAKllvQ9NFLpXRj6aWst4OvPSrQ8s0A_q4'

PERSPECTIVE_API_ATTRIBUTES = {
    'TOXICITY'
}

PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
