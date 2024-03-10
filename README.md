## Masked AutoDecoder
This is the official implementation of the paper "Masked AutoDecoder is Effective Multi-Task Vision Generalist" 

Authors: Han Qiu, Jiaxing Huang, Peng Gao, Lewei Lu, Xiaoqin Zhang, Shijian Lu


In this work, we design Masked AutoDecoder(MAD), an effective multi-task vision generalist. MAD consists of two core designs. First, we develop a parallel decoding framework that introduces bi-directional attention to capture contextual dependencies comprehensively and decode vision task sequences in parallel. Second, we design a masked sequence modeling approach that learns rich task contexts by masking and reconstructing task sequences. In this way, MAD handles all the tasks by a single network branch and a simple cross-entropy loss with minimal task-specific designs.

<div align="center">
  <img src="./figs/architecture.png"/>
</div><br/>