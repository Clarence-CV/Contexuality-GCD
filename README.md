# Contexuality Helps Representation Learning for Generalized Category Discovery


<p align="center">
    <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Wen_Parametric_Classification_for_Generalized_Category_Discovery_A_Baseline_Study_ICCV_2023_paper.html"><img src="https://img.shields.io/badge/-ICCV%202023-68488b"></a>
    <a href="https://arxiv.org/abs/2211.11727"><img src="https://img.shields.io/badge/arXiv-2211.11727-b31b1b"></a>
    <a href="https://wen-xin.info/simgcd"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>
<p align="center">
	Parametric Classification for Generalized Category Discovery: A Baseline Study (ICCV 2023)<br>
  By
  <a href="https://wen-xin.info">Xin Wen</a>*, 
  <a href="https://bzhao.me/">Bingchen Zhao</a>*, and 
  <a href="https://xjqi.github.io/">Xiaojuan Qi</a>.
</p>



This paper introduces a novel approach to Generalized Category Discovery (GCD) by leveraging the concept of contextuality to enhance the identification and classification of categories in unlabeled datasets. Drawing inspiration from human cognition's ability to recognize objects within their context, we propose a dual-context based method. 
		Our model integrates two levels of contextuality: instance-level, where nearest-neighbor contexts are utilized for contrastive learning, and cluster-level, employing prototypical contrastive learning based on category prototypes. The integration of the contextual information effectively improves the feature learning and thereby the classification accuracy of all categories, which better deals with the real-world datasets. Different from the traditional semi-supervised and novel category discovery techniques, our model focuses on a more realistic and challenging scenario where both known and novel categories are present in the unlabeled data.  Extensive experimental results on several benchmark data sets demonstrate that the proposed model outperforms the state-of-the-art. 
## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-10/100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100/1K](https://image-net.org/download.php)


### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

We found picking the model according to 'Old' class performance could lead to possible over-fitting, and since 'New' class labels on the held-out validation set should be assumed unavailable, we suggest not to perform model selection, and simply use the last-epoch model.

## Results
Our results:

<table><thead><tr><th>Source</th><th colspan="3">Paper (3 runs) </th><th colspan="3">Current Github (5 runs) </th></tr></thead><tbody><tr><td>Dataset</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>CIFAR10</td><td>97.1±0.0</td><td>95.1±0.1</td><td>98.1±0.1</td><td>97.0±0.1</td><td>93.9±0.1</td><td>98.5±0.1</td></tr><tr><td>CIFAR100</td><td>80.1±0.9</td><td>81.2±0.4</td><td>77.8±2.0</td><td>79.8±0.6</td><td>81.1±0.5</td><td>77.4±2.5</td></tr><tr><td>ImageNet-100</td><td>83.0±1.2</td><td>93.1±0.2</td><td>77.9±1.9</td><td>83.6±1.4</td><td>92.4±0.1</td><td>79.1±2.2</td></tr><tr><td>ImageNet-1K</td><td>57.1±0.1</td><td>77.3±0.1</td><td>46.9±0.2</td><td>57.0±0.4</td><td>77.1±0.1</td><td>46.9±0.5</td></tr><tr><td>CUB</td><td>60.3±0.1</td><td>65.6±0.9</td><td>57.7±0.4</td><td>61.5±0.5</td><td>65.7±0.5</td><td>59.4±0.8</td></tr><tr><td>Stanford Cars</td><td>53.8±2.2</td><td>71.9±1.7</td><td>45.0±2.4</td><td>53.4±1.6</td><td>71.5±1.6</td><td>44.6±1.7</td></tr><tr><td>FGVC-Aircraft</td><td>54.2±1.9</td><td>59.1±1.2</td><td>51.8±2.3</td><td>54.3±0.7</td><td>59.4±0.4</td><td>51.7±1.2</td></tr><tr><td>Herbarium 19</td><td>44.0±0.4</td><td>58.0±0.4</td><td>36.4±0.8</td><td>44.2±0.2</td><td>57.6±0.6</td><td>37.0±0.4</td></tr></tbody></table>

## Citing this work

If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{wen2023simgcd,
    author    = {Wen, Xin and Zhao, Bingchen and Qi, Xiaojuan},
    title     = {Parametric Classification for Generalized Category Discovery: A Baseline Study},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
    pages     = {16590-16600}
}
```

## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
