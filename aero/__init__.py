"""
AERO: Deep Reinforcement Learning for Account Migration in Sharding Blockchain.

Paper: AERO - Enhancing Sharding Blockchain via Deep Reinforcement Learning
       for Account Migration (WWW 2025).

This package implements the DRL part only. Integration with BlockEmulator
is left for future migration.
"""

from .config import AEROConfig
from .state import AEROState, ShardState
from .action import MigrationTransaction, action_to_migrations, migrations_to_action_list
from .reward import compute_reward, reward_from_state
from .env import AEROEnv
from .network import AEROAttentionPolicy, AEROPolicyValueNet

__all__ = [
    "AEROConfig",
    "AEROState",
    "ShardState",
    "MigrationTransaction",
    "action_to_migrations",
    "migrations_to_action_list",
    "compute_reward",
    "reward_from_state",
    "AEROEnv",
    "AEROAttentionPolicy",
    "AEROPolicyValueNet",
]
