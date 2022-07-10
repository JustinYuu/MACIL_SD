# MACIL_SD
**[ACM MM 2022] Modality-aware Contrastive Instance Learning with Self-Distillation for Weakly-Supervised Audio-Visual Violence Detection**

Jiashuo Yu, Jinyu Liu, Ying Cheng, Rui Feng, Yuejie Zhang  

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

## License

This project is released under the MIT License.

## Acknowledgements  

The codes are based on [XDVioDet](https://github.com/Roc-Ng/XDVioDet) and [RTFM](https://github.com/tianyu0207/RTFM). We sincerely thank them for their efforts. If you have further questions, please contact me at jsyu19@fudan.edu.cn.  
