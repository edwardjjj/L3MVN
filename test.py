import habitat
from habitat.config.default import get_config

config = get_config(config_paths=["./envs/habitat/configs/tasks/objectnav_hm3d.yaml"])


env = habitat.Env(config=config)
