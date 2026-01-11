# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, Any

from omegaconf import DictConfig
from verl.base_config import BaseConfig
from .algorithm import AlgoConfig

__all__ = ["RayConfig", "HybridEngineEntrypointConfig"]

@dataclass
class RayConfig(BaseConfig):
    # Number of CPUs for Ray. Use a fixed number instead of null when using SLURM.
    num_cpus: Optional[int] = None
    # Path to save Ray timeline JSON for performance profiling
    timeline_json_file: Optional[str] = None

@dataclass
class HybridEngineEntrypointConfig(BaseConfig):
    data: Optional[DictConfig] = None
    trainer: Optional[DictConfig] = None
    algorithm: Optional[AlgoConfig] = None
    custom_reward_function: Optional[DictConfig] = None
    ray_init: Optional[RayConfig] = None
    actor_rollout_ref: Optional[DictConfig] = None
    critic: Optional[DictConfig] = None
    reward_model: Optional[DictConfig] = None