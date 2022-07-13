# MACIL_SD  

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/modality-aware-contrastive-instance-learning/anomaly-detection-in-surveillance-videos-on-2)](https://paperswithcode.com/sota/anomaly-detection-in-surveillance-videos-on-2?p=modality-aware-contrastive-instance-learning) 


**【ACM Multimedia 2022】** **Modality-Aware Contrastive Instance Learning with Self-Distillation for Weakly-Supervised Audio-Visual Violence Detection**  


**Authors:** Jiashuo Yu*, Jinyu Liu*, Ying Cheng, Rui Feng, Yuejie Zhang (* equal contribution)  

[Paper](https://arxiv.org/abs/2207.05500)  

> **Abstract:** 
>*Weakly-supervised audio-visual violence detection aims to distinguish snippets containing multimodal violence events with video-level labels. Many prior works perform audio-visual integration and interaction in an early or intermediate manner, yet overlooking the modality heterogeneousness over the weakly-supervised setting. In this paper, we analyze the modality asynchrony and undifferentiated instances phenomena of the multiple instance learning (MIL) procedure, and further investigate its negative impact on weakly-supervised audio-visual learning. To address these issues, we propose a modality-aware contrastive instance learning with self-distillation (MACIL-SD) strategy . Specifically, we leverage a lightweight two-stream network to generate audio and visual bags, in which unimodal background, violent, and normal instances are clustered into semi-bags in an unsupervised way. Then audio and visual violent semi-bag representations are assembled as positive pairs, and violent semi-bags are combined with background and normal instances in the opposite modality as contrastive negative pairs. Furthermore, a self-distillation module is applied to transfer unimodal visual knowledge to the audio-visual model, which alleviates noises and closes the semantic gap between unimodal and multimodal features. Experiments show that our framework outperforms previous methods with lower complexity on the large-scale XD-Violence dataset. Results also demonstrate that our proposed approach can be used as plug-in modules to enhance other networks.* 


**Overview**

<p align="center">
    <img src=overview.png width="800" height="300"/>
</p>

**Comparison with SOTAs.**

| Method    | Modality |AP (%) | Params |
| ----------| :------: | :----:| :----: |
| Sultani et al. | Visual | 73.20 | / |
| RTFM | Visual | 77.81 |12.067M |
| Li et al. | Visual | 78.28 | /|
| Wu et al. | Audio+Visuial | 78.64 |1.539M |
| Pang et al.| Audio+Visual | 81.69 | 1.876M|
| Ours (light)| Audio+Visual | **82.17** | **0.347M**|
| Ours (full)| Audio+Visual | **83.40** | **0.678M**|

## XD-Violence Dataset & Features  

The audio and visual features of the XD-Violence dataset can be downloaded at this [link](https://roc-ng.github.io/XD-Violence/). Note that in this paper, only the **RGB** and **VGGish** features are required. You can download the **RGB.zip**, **RGBTest.zip**, and **vggish-features.zip** and unzip them into the *data/* folder.  

## Requirements  

    python==3.7.11  
    torch==1.6.0  
    cuda==10.1  
    numpy==1.17.4
  
Note that the reported results are obtained by training on a single Tesla V100 GPU. We observe that different GPU types and torch/cuda versions can lead to slightly different results.  

## Training

`python main.py --model_name=macil_sd`  

## Testing

`python infer.py  --model_dir=macil_sd.pkl`  

## Citation  

If you find our work interesting and useful, please consider citing it.  

    @article{yu2022macil,
      title={Modality-Aware Contrastive Instance Learning with Self-Distillation for Weakly-Supervised Audio-Visual Violence Detection},
      author={Jiashuo Yu, Jinyu Liu, Ying Cheng, Rui Feng, Yuejie Zhang},
      journal={arXiv preprint arXiv:2207.05500},
      year={2022}
    }  

## License

This project is released under the MIT License.

## Acknowledgements  

The codes are based on [XDVioDet](https://github.com/Roc-Ng/XDVioDet) and [RTFM](https://github.com/tianyu0207/RTFM). We sincerely thank them for their efforts. If you have further questions, please contact us at jsyu19@fudan.edu.cn and jinyuliu20@fudan.edu.cn.  
