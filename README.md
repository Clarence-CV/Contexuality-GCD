# Contexuality Helps Representation Learning for Generalized Category Discovery






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


## Acknowledgements

The codebase is largely built on this repo: https://github.com/sgvaze/generalized-category-discovery.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
