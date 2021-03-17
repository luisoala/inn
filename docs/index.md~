<!-- mathjax include -->
<!-- {% include mathjax.html %} -->
<!-- defining some tex commands that can be used throughout the page-->
# Abstract
In this work, we propose MixMOOD - a systematic approach to mitigate the effects of class distribution mismatch in semi-supervised deep learning (SSDL) with MixMatch. This work is divided into two components: (i) an extensive out of distribution (OOD) ablation test bed for SSDL and (ii) a quantitative unlabelled dataset selection heuristic referred to as MixMOOD. In the first part, we analyze the sensitivity of MixMatch accuracy under 90 different distribution mismatch scenarios across three multi-class classification tasks. These are designed to systematically understand how OOD unlabelled data affects MixMatch performance. In the second part, we propose an efficient and effective method, called deep dataset dissimilarity measures (DeDiMs), to compare labelled and unlabelled datasets. The proposed DeDiMs are quick to evaluate and model agnostic. They use the feature space of a generic Wide-ResNet and can be applied prior to learning. Our test results reveal that supposed semantic similarity between labelled and unlabelled data is not a good heuristic for unlabelled data selection. In contrast, strong correlation between MixMatch accuracy and the proposed DeDiMs allow us to quantitatively rank different unlabelled datasets ante hoc according to expected MixMatch accuracy. This is what we call MixMOOD. Furthermore, we argue that the MixMOOD approach can aid to standardize the evaluation of different semi-supervised learning techniques under real world scenarios involving out of distribution data.
<!--# OOD and IOD-->
<!--TODO: definitions-->
# Highlights and Findings
We present the MixMatch approach to out of distribution data (MixMOOD). It entails the following contributions:
* A systematic OOD ablation test bed. We demonstrate that including OOD data in the unlabelled training dataset for the MixMatch algorithm can yield different degrees of accuracy degradation compared to the exclusive use of IOD data. However, in most cases, using unlabelled data with OOD contamination still improves the results when compared to the default fully supervised configuration.
* Markedly, OOD data that is supposedly semantically similar to the OOD labelled data does not always lead to the highest accuracy gain.
* We propose and evaluate four deep dataset dissimilarity measures (DeDiMs) that can be used to rank unlabelled data according to the expected accuracy gain _prior_ to SSDL training. They are cheap to compute and model-agnostic which make them amenable for practical application.
* Our test results reveal a strong correlation between the tested DeDiMs and MixMatch accuracy, making them informative for unlabelled dataset selection. This leads to MixMOOD which proposes the usage of tested \gls{DeDiM}s to select the unlabelled dataset for improved MixMatch accuracy.

<!--![Table 1](https://github.com/luisoala/mixmood/blob/master/docs/imgs/table1.png)-->
<!--<img src="https://luisoala.github.io/mixmood/imgs/table1.png" style="display: block; margin: auto;" /> -->
<!--![Table 3](https://github.com/luisoala/mixmood/blob/master/docs/imgs/table3.png)-->
<!--!# Recommendations and Closing Thoughts
TODO: closing thoughs-->
# Data Used
All data can be found [here](https://drive.google.com/drive/folders/1CFJdMxcX96BGJ2L7cotLAJNKWcpZba7k?usp=sharing). Note that when you use our bash scripts to reproduce experiments the necessary data will automatically be downloaded. Below you can find a short summary of the datasets.
![useful image](./assets/data-table.png) 
<!-- <img src="{{ site.url }}/docs/assets/table1.png" style="display: block; margin: auto;" /> -->
# Questions?
If you have questions regarding the code or the paper or if would like to discuss ideas please open an issue in the [repo](https://github.com/luisoala/mixmood).

<!--TODO port to the other three papers -> start with ICML-->


