import habitat

config = habitat.get_config(
    config_path="./envs/habitat/configs/tasks/objectnav_hm3d.yaml"
)

env = habitat.Env(config=config)
