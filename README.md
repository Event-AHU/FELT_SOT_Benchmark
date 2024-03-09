# FELT_SOT_Benchmark 

<div align="center">

<img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/figures/FELT_SOT_logo.png" width="600">
  
**The First Frame-Event Long-Term Single Object Tracking Benchmark** 

------

<p align="center">
  • <a href="">arXiv</a> • 
  <a href="">Baselines</a> •
  <a href="">DemoVideo</a> • 
  <a href="">Tutorial</a> •
</p>

</div>


# :dart: Abstract 
Current event-/frame-event based trackers undergo evaluation on short-term tracking datasets, however, the tracking of real-world scenarios involves long-term tracking, and the performance of existing tracking algorithms in these scenarios remains unclear. In this paper, we first propose a new long-term and large-scale frame-event single object tracking dataset, termed FELT. It contains 742 videos and 1,594,474 RGB frames and event stream pairs and has become the largest frame-event tracking dataset to date. We re-train and evaluate 15 baseline trackers on our dataset for future works to compare. More importantly, we find that the RGB frames and event streams are naturally incomplete due to the influence of challenging factors and spatially sparse event flow. In response to this, we  propose a novel associative memory Transformer network as a unified backbone by introducing modern Hopfield layers into multi-head self-attention blocks to fuse both RGB and event data. Extensive experiments on both FELT and RGB-T tracking dataset LasHeR fully validated the effectiveness of our model. 




# :collision: Update Log 



# :video_camera: Demo Video 
* **The demo video for the FELT SOT dataset is available on** [[**Youtube**](https://youtu.be/6zxiBHTqOhE?si=6ARRGFdBLSxyp3G8)]. 

<p align="center">
  <a href="https://youtu.be/6zxiBHTqOhE?si=6ARRGFdBLSxyp3G8">
    <img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/figures/felt_sot_demo_screenshot.png" alt="DemoVideo" width="700"/>
  </a>
</p>







# :hammer: Environment






## Framework 
<p align="center">
<img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/figures/frameworkV2.jpg" alt="framework" width="700"/>
</p>

Install env
```
conda create -n hdetrack python=3.8
conda activate hdetrack
bash install.sh
```

Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```



## Train & Test & Evaluation










## Test FLOPs, and Speed





# :dvd: FELT_SOT Dataset 

<p align="center">
<img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/figures/felt_sot_samples.jpg" alt="felt_sot_samples" width="700"/>
</p>

* **BaiduYun:** 
```
https://pan.baidu.com/s/12ur7n1wSDvIWajPQJMd8Kg?pwd=AHUT 
```

* **DropBox:**
```
https://www.dropbox.com/scl/fo/0n5m12gt30drsha30hgth/h?rlkey=20mpz2oh1etbv8cnsav01bhj5&dl=0
```

The directory should have the below format:
```Shell
├── FELT_SOT dataset
    ├── Training Subset (520 videos, 470.23GB)
        ├── dvSave-2022_10_11_19_24_36
            ├── dvSave-2022_10_11_19_24_36_aps
            ├── dvSave-2022_10_11_19_24_36_dvs
            ├── dvSave-2022_10_11_19_24_36.aedat4
            ├── groundtruth.txt
            ├── absent.txt
        ├── ... 
    ├── Testing Subset (222 videos, 194.93GB)
        ├── dvSave-2022_10_11_19_43_03
            ├── dvSave-2022_10_11_19_43_03_aps
            ├── dvSave-2022_10_11_19_43_03_dvs
            ├── dvSave-2022_10_11_19_43_03_dvs.aedat4
            ├── groundtruth.txt
            ├── absent.txt
        ├── ...
```

# :triangular_ruler: Evaluation Toolkit






# :chart_with_upwards_trend: Benchmark Results 
<p align="center">
<img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/figures/FELT_results.jpg" alt="framework" width="900"/>
</p>

# :cupid: Acknowledgement 
[[CEUTrack](https://github.com/Event-AHU/COESOT)] 
[[VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)] 
[[FE108](https://github.com/Jee-King/ICCV2021_Event_Frame_Tracking)] 
[[Awesome_Modern_Hopfield_Networks](https://github.com/Event-AHU/Awesome_Modern_Hopfield_Networks)] 
[[OSTrack](https://github.com/botaoye/OSTrack)] 

# :newspaper: Citation 
If you think this project is helpful, please feel free to leave a star ⭐️ and cite our paper:

```bibtex
@article{wang2023feltsot,
  title={Event Camera based Long-term Visual Tracking: A Benchmark},
  author={},
  journal={},
  year={2023}
}
```




