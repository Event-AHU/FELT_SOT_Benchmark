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


# :collision: Update Log 

<!-- * [2025.08.10] The FELT SOT dataset, baseline, benchmarked results, and evaluation toolkit are all released. -->
* [2025.08.06] Our arXiv paper is available at [[arXiv](https://arxiv.org/pdf/2403.05839v3)]. 
<!-- latest version -->



# :hammer: Environment

## Framework 
<p align="center">
<img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/AMTTrack_v2/figures/framework.jpg" alt="framework" width="700"/>
</p>


* **BaiduYun:** 
```
FELT V2：TO BE UPDATED 
```
<!-- FELT V2：https://pan.baidu.com/s/1AiUTsvvsCKj8lWuc-821Eg?pwd=AHUT -->

* **DropBox:**
```
FELT V2：TO BE UPDATED 
```

The directory should have the below format:
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