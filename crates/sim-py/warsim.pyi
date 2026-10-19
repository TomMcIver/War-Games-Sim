"""Type stubs for the `warsim` extension module (see crates/sim-py/README.md)."""

from types import TracebackType
from typing import Final, Literal, TypedDict

import numpy as np
import numpy.typing as npt

__version__: Final[str]
TOKEN_FEATS: Final[int]
MAP_CHANNELS: Final[int]
LOCAL_VIEW: Final[int]
STATE_FEATS: Final[int]
MAX_TEAM_SIZE: Final[int]
MAX_UNITS: Final[int]
ACTION_HEADS: Final[int]
ACTION_DIMS: Final[tuple[int, int, int, int, int]]
MAP_KINDS: Final[tuple[str, ...]]
SCENARIOS: Final[tuple[str, ...]]

class Obs(TypedDict):
    tokens: npt.NDArray[np.float32]  # [E, U, U, TOKEN_FEATS]
    token_mask: npt.NDArray[np.float32]  # [E, U, U]
    local_map: npt.NDArray[np.float32]  # [E, U, MAP_CHANNELS, LOCAL_VIEW, LOCAL_VIEW]
    state: npt.NDArray[np.float32]  # [E, U * STATE_FEATS]
    alive: npt.NDArray[np.float32]  # [E, U]
    action_availability: npt.NDArray[np.float32]  # [E, U, 2] shoot, interact

class Infos(TypedDict):
    winner: npt.NDArray[np.int8]  # [E], -1 draw/none, else team id; valid where dones
    episode_len: npt.NDArray[np.uint32]  # [E], valid where dones
    episode_stats: dict[str, npt.NDArray]  # named [E] arrays; zero on non-terminal steps

class VecEnv:
    def __init__(
        self,
        n_envs: int,
        team_size: int = 2,
        map_kind: Literal["arena", "building", "village"] | str = "arena",
        max_ticks: int = 600,
        seed: int = 0,
        tick_seconds: float = 0.1,
        num_threads: int | None = None,
        *,
        scenario: Literal["standard", "duel", "cover", "door", "skirmish"] | str = "standard",
        reward_win: float | None = None,
        reward_loss: float | None = None,
        reward_draw: float | None = None,
        reward_damage_dealt: float | None = None,
        reward_kill: float | None = None,
        reward_damage_taken: float | None = None,
        reward_time_penalty: float | None = None,
        reward_ammo_waste: float | None = None,
        shaping_scale: float | None = None,
    ) -> None: ...
    def reset(self) -> Obs: ...
    def step(
        self, actions: npt.NDArray[np.uint8]
    ) -> tuple[Obs, npt.NDArray[np.float32], npt.NDArray[np.bool_], Infos]: ...
    def close(self) -> None: ...
    def __enter__(self) -> VecEnv: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool: ...
    @property
    def n_envs(self) -> int: ...
    @property
    def n_units(self) -> int: ...
    @property
    def team_size(self) -> int: ...
    @property
    def action_dims(self) -> list[int]: ...
    @property
    def closed(self) -> bool: ...
