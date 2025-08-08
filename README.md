# FELT_SOT_Benchmark 

<div align="center">

<img src="https://github.com/Event-AHU/FELT_SOT_Benchmark/blob/main/figures/FELT_SOT_logo.png" width="600">
  
**The First Frame-Event Long-Term Single Object Tracking Benchmark** 

------
<!-- 
<p align="center">
  • <a href="https://arxiv.org/abs/2403.05839">arXiv</a> • 
  <a href="https://github.com/Event-AHU/FELT_SOT_Benchmark">Code</a> •
  <a href="https://youtu.be/6zxiBHTqOhE?si=6ARRGFdBLSxyp3G8">DemoVideo</a> • 
  <a href="">Tutorial</a> •
</p> -->

</div>

> **Long-Term Visual Object Tracking with Event Cameras: An Associative Memory Augmented Tracker and A Benchmark Dataset**, Xiao Wang, Xufeng Lou, Shiao Wang, Ju Huang, Lan Chen, Bo Jiang, arXiv:2403.05839
[[arXiv]](https://arxiv.org/abs/2403.05839)
[[Paper](https://arxiv.org/pdf/2403.05839)] 
[[Code](https://github.com/Event-AHU/FELT_SOT_Benchmark)] 
[[DemoVideo](https://youtu.be/6zxiBHTqOhE?si=6ARRGFdBLSxyp3G8)]  


# :dart: Abstract 
Existing event stream based trackers undergo evaluation on short-term tracking datasets, however, the tracking of real-world scenarios involves long-term tracking, and the performance of existing tracking algorithms in these scenarios remains unclear. In this paper, we first propose a new long-term, large-scale frame-event visual object tracking dataset, termed FELT. It contains 1,044 long-term videos that involve 1.9 million RGB frames and event stream pairs, 60 different target objects, and 14 challenging attributes. To build a solid benchmark, we retrain and evaluate 21 baseline trackers on our dataset for future work to compare. In addition, we propose a novel Associative Memory Transformer based RGB-Event long-term visual tracker, termed AMTTrack. It follows a one-stream tracking framework and aggregates the multi-scale RGB/event template and search tokens effectively via the Hopfield retrieval layer. The framework also embodies another aspect of associative memory by maintaining dynamic template representations through an associative memory update scheme, which addresses the appearance variation in long-term tracking. Extensive experiments on FELT, FE108, VisEvent, and COESOT datasets fully validated the effectiveness of our proposed tracker.


