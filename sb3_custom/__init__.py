import os

from sb3_custom.ars import ARS
# from sb3_custom.ppo_mask import MaskablePPO
from sb3_custom.ppo_recurrent import RecurrentPPO
from sb3_custom.qrdqn import QRDQN
from sb3_custom.tqc import TQC
from sb3_custom.trpo import TRPO

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()
