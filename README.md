# An Image is Worth Flexible Tokens like Sentences



## Trian the FlexTok
```
accelerate launch --num_machines=1 --num_processes=4 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network train_flextitok.py config=configs/training/FlexTok/stage1/imagenet_s128.yaml
```