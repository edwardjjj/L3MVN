1. Assume working directory is l3mvn
Download datasets
```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_v0.1 --data-path l3mvn/data
```


2. uncomment code in envs/habitat/objectgoal_env21.py
```semantic = np.expand_dims(semantic.astype(np.uint8), 2)```

3. Download segmentation model from [here](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link). Put the downloaded file in l3mvn/Rednet/model

4. Download test set from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip). Unzip and rename the folder to objectgoal_hm3d and place it in l3mvn/data

5. Build Docker image
```
docker build -t l3mvn:1.0 .
```
6. Run the image
```
docker run --gpus all -v .:/app/l3mvn -v -it l3mvn:1.0
```
inside the container run the following to test the feed-forward method
```
cd l3mvn
. activate habitat
python main_llm_vis.py --split val --eval 1 --auto_gpu_config 0 \
-n 8 --num_eval_episodes 250 --load pretrained_models/llm_model.pt \
--use_gtsem 0 --num_local_steps 10
```
run the following to test the zero-shot method
```
cd l3mvn
. activate habitat
python main_llm_zeroshot.py --split val --eval 1 --auto_gpu_config 0 \
-n 5 --num_eval_episodes 400 --num_processes_on_first_gpu 5 \
--use_gtsem 0 --num_local_steps 10 --exp_name exp_llm_hm3d_zero \
