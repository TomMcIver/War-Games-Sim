"""Shape constants mirrored from crates/sim-core/src/types.rs (see docs/INTERFACE.md)."""

TOKEN_FEATS = 12
MAP_CHANNELS = 4
LOCAL_VIEW = 9
STATE_FEATS = 10
MAX_TEAM_SIZE = 8
MAX_UNITS = 2 * MAX_TEAM_SIZE
ACTION_DIMS = (3, 3, 3, 2, 2)
ACTION_HEADS = len(ACTION_DIMS)

OBS_KEYS = ("tokens", "token_mask", "local_map", "state", "alive")

# Winner codes carried in infos["winner"].
DRAW = -1
