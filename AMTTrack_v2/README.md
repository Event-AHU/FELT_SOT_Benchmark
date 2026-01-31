# FELT_SOT_Benchmark 

<div align="center">
<!-- ------ -->
</div>

> **Long-Term Visual Object Tracking with Event Cameras: An Associative Memory Augmented Tracker and A Benchmark Dataset**, Xiao Wang, Xufeng Lou, Shiao Wang, Ju Huang, Lan Chen, Bo Jiang, arXiv:2403.05839
[[arXiv]](https://arxiv.org/abs/2403.05839)
[[Paper](https://arxiv.org/pdf/2403.05839)] 
[[Code](https://github.com/Event-AHU/FELT_SOT_Benchmark)] 
[[DemoVideo](https://youtu.be/6zxiBHTqOhE?si=6ARRGFdBLSxyp3G8)]  


# :dart: Abstract 
Existing event stream based trackers undergo evaluation on short-term tracking datasets, however, the tracking of real-world scenarios involves long-term tracking, and the performance of existing tracking algorithms in these scenarios remains unclear. In this paper, we first propose a new long-term, large-scale frame-event visual object tracking dataset, termed FELT. It contains 1,044 long-term videos that involve 1.9 million RGB frames and event stream pairs, 60 different target objects, and 14 challenging attributes. To build a solid benchmark, we retrain and evaluate 21 baseline trackers on our dataset for future work to compare. In addition, we propose a novel Associative Memory Transformer based RGB-Event long-term visual tracker, termed AMTTrack. It follows a one-stream tracking framework and aggregates the multi-scale RGB/event template and search tokens effectively via the Hopfield retrieval layer. The framework also embodies another aspect of associative memory by maintaining dynamic template representations through an associative memory update scheme, which addresses the appearance variation in long-term tracking. Extensive experiments on FELT, FE108, VisEvent, and COESOT datasets fully validated the effectiveness of our proposed tracker.

## Framework 
<p align="center">
<img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/AMTTrack_v2/figures/framework.jpg" alt="framework" width="700"/>
</p>

# :collision: Update Log 

<!-- * [2026.01.31] The FELT SOT dataset, baseline, benchmarked results, and evaluation toolkit are all released. -->
* [2026.01.31] The FELT SOT dataset, baseline, benchmarked results, and evaluation toolkit are all released.
* [2025.08.06] Our arXiv paper is available at [[arXiv](https://arxiv.org/pdf/2403.05839v3)]. 



# :hammer: Environment

### Install environment
```
conda create -n amttrack python=3.8
conda activate amttrack
bash install.sh
```

### Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### Data preparation
* Then, put the tracking datasets FELT in `./data` from [[Baidu Drive]](https://pan.baidu.com/s/1AiUTsvvsCKj8lWuc-821Eg?pwd=AHUT). It should look like:
    ```Shell
    ├── FELT_SOT dataset
        ├── Training Subset (730 videos)
            ├── dvSave-2022_10_11_19_24_36
                ├── dvSave-2022_10_11_19_24_36_aps
                ├── dvSave-2022_10_11_19_24_36_dvs
                ├── groundtruth.txt
                ├── absent.txt
            ├── ... 
        ├── Testing Subset (314 videos)
            ├── dvSave-2022_10_11_19_43_03
                ├── dvSave-2022_10_11_19_43_03_aps
                ├── dvSave-2022_10_11_19_43_03_dvs
                ├── groundtruth.txt
                ├── absent.txt
            ├── ...
    ```

* Download pre-trained models from [[Baidu Drive]](https://pan.baidu.com/s/1X3pzbV07lCIsclsg_iNW9w?pwd=AHUT) and put it under `$/pretrained_networks` for training.

* Download the trained model weights from [[Baidu Drive]](https://pan.baidu.com/s/1_orgI_BvzWZP-4ujtQIOnQ?pwd=AHUT) and put it under `$/output/checkpoints/train/amttrack/felt/` for testing directly.

* Note: DropBox will be updated.

# Train & Test

```
# train
python tracking/train.py --script amttrack --config felt --save_dir ./output --mode single --use_wandb 0

# test
python tracking/test.py --tracker_name amttrack --tracker_param felt --dataset_name felt --threads 0 --num_gpus 1
```

<!-- ## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX 2080Ti GPU. -->

# Evaluation Toolkit

1. Download the FELTv2_eval_toolkit from [[Baidu Drive]](https://pan.baidu.com/s/1AiUTsvvsCKj8lWuc-821Eg?pwd=AHUT), and open it with Matlab (over Matlab R2020).
2. add your tracking results in `$data/felt_tracking_results/` and modify the name in `$/utils/config_tracker.m`
3. run `Evaluate_FELT_benchmark_SP_PR_NPR.m` for the overall performance evaluation, including SR, PR, NPR. 
Modify the value of the_type, where 1 corresponds to SR, 2 corresponds to PR, and 3 corresponds to NPR.


# :cupid: Acknowledgement 
[[CEUTrack](https://github.com/Event-AHU/COESOT)] 
[[VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)] 
[[FE108](https://github.com/Jee-King/ICCV2021_Event_Frame_Tracking)] 
[[Awesome_Modern_Hopfield_Networks](https://github.com/Event-AHU/Awesome_Modern_Hopfield_Networks)] 
[[OSTrack](https://github.com/botaoye/OSTrack)] 

# :newspaper: Citation 
If you think this project is helpful, please feel free to leave a star ⭐️ and cite our paper:

```bibtex
@misc{wang2025longtermvisualobjecttracking,
      title={Long-Term Visual Object Tracking with Event Cameras: An Associative Memory Augmented Tracker and A Benchmark Dataset}, 
      author={Xiao Wang and Xufeng Lou and Shiao Wang and Ju Huang and Lan Chen and Bo Jiang},
      year={2025},
      eprint={2403.05839},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.05839}, 
}
```