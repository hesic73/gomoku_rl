import os

CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")
)
from .env import GomokuEnv
