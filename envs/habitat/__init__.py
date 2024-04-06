# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os

import numpy as np
import torch
from habitat import Env, RLEnv, make_dataset
from habitat.config import read_write
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from agents.sem_exp import Sem_Exp_Env_Agent

from .objectgoal_env import ObjectGoal_Env
from .objectgoal_env21 import ObjectGoal_Env21
from .utils.vector_env import VectorEnv


def make_env_fn(args, config_env, rank):
    dataset = make_dataset(config_env.dataset.type, config=config_env.dataset)
    with read_write(config_env):
        config_env.simulator.scene = dataset.episodes[0].scene_id

    if args.agent == "sem_exp":
        env = Sem_Exp_Env_Agent(
            args=args, rank=rank, config_env=config_env, dataset=dataset
        )
    elif args.agent == "obj21":
        env = ObjectGoal_Env21(
            args=args, rank=rank, config_env=config_env, dataset=dataset
        )
    else:
        env = ObjectGoal_Env(
            args=args, rank=rank, config_env=config_env, dataset=dataset
        )

    env.seed(rank)
    return env


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".glb.json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename[: -len(scene_dataset_ext) + 4]
            scenes.append(scene)
    scenes.sort()
    return scenes


def construct_envs(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_path="configs/" + args.task_config)
    if "habitat" in basic_config:
        basic_config = basic_config.habitat
    basic_config.dataset.split = args.split
    # basic_config.DATASET.DATA_PATH = \
    #     basic_config.DATASET.DATA_PATH.replace("v1", args.version)
    # basic_config.DATASET.EPISODES_DIR = \
    #     basic_config.DATASET.EPISODES_DIR.replace("v1", args.version)

    scenes = basic_config.dataset.content_scenes
    if "*" in basic_config.dataset.content_scenes:
        content_dir = os.path.join(
            "data/datasets/objectnav/gibson/v1.1/" + args.split, "content"
        )
        scenes = _get_scenes_from_folder(content_dir)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there " "aren't enough number of scenes"
        )

        scene_split_sizes = [
            int(np.floor(len(scenes) / args.num_processes))
            for _ in range(args.num_processes)
        ]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    for i in range(args.num_processes):
        config_env = cfg_env(config_path="configs/" + args.task_config)
        if "habitat" in config_env:
            config_env = config_env.habitat

        if len(scenes) > 0:
            config_env.dataset.content_scenes = scenes[
                sum(scene_split_sizes[:i]) : sum(scene_split_sizes[: i + 1])
            ]
            print("Thread {}: {}".format(i, config_env.dataset.content_scenes))

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = (
                int((i - args.num_processes_on_first_gpu) // args.num_processes_per_gpu)
                + args.sim_gpu_id
            )
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.simulator.habitat_sim_v0.gpu_device_id = gpu_id

        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")
        agent_sensors.append("SEMANTIC_SENSOR")

        config_env.simulator.agent_0.sensors = agent_sensors

        # Reseting episodes manually, setting high max episode length in sim
        config_env.environment.max_episode_steps = 10000000
        config_env.environment.iterator_options.shuffle = False

        # config_env.simulator.rgb_sensor.width = args.env_frame_width
        # config_env.simulator.rgb_sensor.height = args.env_frame_height
        # config_env.simulator.rgb_sensor.hfov = args.hfov
        # config_env.simulator.rgb_sensor.position = [0, args.camera_height, 0]
        #
        # config_env.simulator.depth_sensor.width = args.env_frame_width
        # config_env.simulator.depth_sensor.height = args.env_frame_height
        # config_env.simulator.depth_sensor.hfov = args.hfov
        # config_env.simulator.depth_sensor.min_depth = args.min_depth
        # config_env.simulator.depth_sensor.max_depth = args.max_depth
        # config_env.simulator.depth_sensor.position = [0, args.camera_height, 0]
        #
        # config_env.simulator.semantic_sensor.width = args.env_frame_width
        # config_env.simulator.semantic_sensor.height = args.env_frame_height
        # config_env.simulator.semantic_sensor.hfov = args.hfov
        # config_env.simulator.semantic_sensor.position = [0, args.camera_height, 0]

        config_env.simulator.turn_angle = args.turn_angle
        config_env.dataset.split = args.split
        # config_env.DATASET.DATA_PATH = \
        #     config_env.DATASET.DATA_PATH.replace("v1", args.version)
        # config_env.DATASET.EPISODES_DIR = \
        #     config_env.DATASET.EPISODES_DIR.replace("v1", args.version)

        config_env.freeze()
        env_configs.append(config_env)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(args_list, env_configs, range(args.num_processes)))
        ),
    )

    return envs


def construct_envs21(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_path="configs/" + args.task_config)
    if "habitat" in basic_config:
        basic_config = basic_config.habitat
    with read_write(basic_config):
        basic_config.dataset.split = args.split
        basic_config.dataset.data_path = basic_config.dataset.data_path.replace(
            "v1", args.version
        )
    # basic_config.DATASET.EPISODES_DIR = \
    #     basic_config.DATASET.EPISODES_DIR.replace("v1", args.version)

    scenes = basic_config.dataset.content_scenes
    dataset = make_dataset(basic_config.dataset.type, config=basic_config.dataset)
    if "*" in basic_config.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(basic_config.dataset)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there " "aren't enough number of scenes"
        )

        scene_split_sizes = [
            int(np.floor(len(scenes) / args.num_processes))
            for _ in range(args.num_processes)
        ]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    for i in range(args.num_processes):
        config_env = cfg_env(config_path="configs/" + args.task_config)
        if "habitat" in config_env:
            config_env = config_env.habitat

        if len(scenes) > 0:
            with read_write(config_env):
                config_env.dataset.content_scenes = scenes[
                    sum(scene_split_sizes[:i]) : sum(scene_split_sizes[: i + 1])
                ]
                print("Thread {}: {}".format(i, config_env.dataset.content_scenes))

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = (
                int((i - args.num_processes_on_first_gpu) // args.num_processes_per_gpu)
                + args.sim_gpu_id
            )
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        with read_write(config_env):
            config_env.simulator.habitat_sim_v0.gpu_device_id = gpu_id

            # Reseting episodes manually, setting high max episode length in sim
            config_env.environment.max_episode_steps = 10000000
            config_env.environment.iterator_options.shuffle = False

            # config_env.simulator.rgb_sensor.width = args.env_frame_width
            # config_env.simulator.rgb_sensor.height = args.env_frame_height
            # config_env.simulator.rgb_sensor.hfov = args.hfov
            # config_env.simulator.rgb_sensor.position = [0, args.camera_height, 0]
            #
            # config_env.simulator.depth_sensor.width = args.env_frame_width
            # config_env.simulator.depth_sensor.height = args.env_frame_height
            # config_env.simulator.depth_sensor.hfov = args.hfov
            # config_env.simulator.depth_sensor.min_depth = args.min_depth
            # config_env.simulator.depth_sensor.max_depth = args.max_depth
            # config_env.simulator.depth_sensor.position = [0, args.camera_height, 0]
            #
            # config_env.simulator.semantic_sensor.width = args.env_frame_width
            # config_env.simulator.semantic_sensor.height = args.env_frame_height
            # config_env.simulator.semantic_sensor.hfov = args.hfov
            # config_env.simulator.semantic_sensor.position = [0, args.camera_height, 0]

            config_env.simulator.turn_angle = args.turn_angle
            config_env.dataset.split = args.split
            config_env.dataset.data_path = config_env.dataset.data_path.replace(
                "v1", args.version
            )
        # config_env.DATASET.EPISODES_DIR = \
        #     config_env.DATASET.EPISODES_DIR.replace("v1", args.version)

        env_configs.append(config_env)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(args_list, env_configs, range(args.num_processes)))
        ),
    )

    return envs
