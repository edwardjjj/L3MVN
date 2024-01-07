import habitat
from habitat.config.default import get_config

config = get_config(config_path="objectnav.yaml")
env = habitat.Env(config=config)
obs = env.reset()
print(obs)
