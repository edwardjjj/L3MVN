import habitat
from habitat.config.default import get_config

config = get_config(config_path="objectnav.yaml")
env = habitat.RLEnv(config=config)
obs, info = env.reset()
print(obs)
