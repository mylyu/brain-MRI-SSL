# brain-MRI-SSL

Official repository for the paper **"Tailored self-supervised pretraining improves brain MRI diagnostic models"**, published in *Computerized Medical Imaging and Graphics*.

<br>

# Models & Data
The **key model checkpoints** and validation datasets from our paper are available for download from a single location in **GitHub Releases**.

### **[>> Go to GitHub Releases to Download Model Checkpoints <<](https://github.com/mylyu/brain-MRI-SSL/releases)**

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/mylyu/brain-MRI-SSL?style=for-the-badge&label=Latest%20Release)](https://github.com/mylyu/brain-MRI-SSL/releases)

<br>

We provide four key checkpoints:

1.  **MoCo-v3 (4-Class):** `mocov3-brain4class.ckpt`
    -   *Corresponds to the main result in **Table 2** of our paper ("Tailored" pretraining), achieving **76.14%** accuracy.*

2.  **MoCo-v3 (16-Class):** `mocov3-brain16class.ckpt`
    -   *Corresponds to the main result in **Table 3** of our paper, achieving **90.98%** accuracy.*

3.  **SimCLR (4-Class & 16-Class)\***:
    -   `simclr-brain4class.ckpt` & `simclr-brain16class.ckpt`
    -   *\*Note: These models were trained on the same tailored datasets. Post-publication analysis revealed that they achieve even better performance than the MoCo-v3 models.*

# Publication
If you find our work useful in your research or publication, please cite our work:

[1] Huang, X., Wang, Z., Zhou, W. et al. Tailored self-supervised pretraining improves brain MRI diagnostic models. *Comput Med Imaging Graph* **123**, 102560 (2025). https://doi.org/10.1016/j.compmedimag.2025.102560

```bibtex
@article{huang_tailored_2025,
  title = {Tailored self-supervised pretraining improves brain {MRI} diagnostic models},
  volume = {123},
  issn = {0895-6111},
  url = {https://doi.org/10.1016/j.compmedimag.2025.102560},
  doi = {10.1016/j.compmedimag.2025.102560},
  journal = {Computerized Medical Imaging and Graphics},
  author = {Huang, Xinhao and Wang, Zihao and Zhou, Weichen and Yang, Kexin and Wen, Kaihua and Liu, Haiguang and Huang, Shoujin and Lyu, Mengye},
  year = {2025},
  pages = {102560},
}
```

# Abstract
Self-supervised learning has shown potential in enhancing deep learning methods, yet its application in brain magnetic resonance imaging (MRI) analysis remains underexplored. This study seeks to leverage large-scale, unlabeled public brain MRI datasets to improve the performance of deep learning models in various downstream tasks for the development of clinical decision support systems. To enhance training efficiency, data filtering methods based on image entropy and slice positions were developed, condensing a combined dataset of approximately 2 million images from fastMRI-brain, OASIS-3, IXI, and BraTS21 into a more focused set of 250 K images enriched with brain features. The Momentum Contrast (MoCo) v3 algorithm was then employed to learn these image features, resulting in robustly pretrained models specifically tailored to brain MRI. The pretrained models were subsequently evaluated in tumor classification, lesion detection, hippocampal segmentation, and image reconstruction tasks. The results demonstrate that our brain MRI-oriented pretraining outperformed both ImageNet pretraining and pretraining on larger multi-organ, multi-modality medical datasets, achieving a ∼2.8 % increase in 4-class tumor classification accuracy, a ∼0.9 % improvement in tumor detection mean average precision, a ∼3.6 % gain in adult hippocampal segmentation Dice score, and a ∼0.1 PSNR improvement in reconstruction at 2-fold acceleration. This study underscores the potential of self-supervised learning for brain MRI using large-scale, tailored datasets derived from public sources.

<p align="center">
  <img src="https://github.com/user-attachments/assets/eefa2a54-0c2b-4276-b7ce-c50825888b6c" width="800px" />
  <br>
</p>

# Usage
This repository provides a single Python script (`evaluation.py`) for reproducing the classification results.

### Quick Start
1.  **Clone this Repository.**
    ```bash
    git clone https://github.com/mylyu/brain-MRI-SSL.git
    cd brain-MRI-SSL
    ```
2.  **Download Files from Release.**
*   Go to our **[Releases](https://github.com/mylyu/brain-MRI-SSL/releases)** page.
*   Download `datasets.zip` and unzip it. You should now have a `datasets/` folder in your project root.
*   Download the model checkpoints and place them in a folder of your choice (e.g., `models/`).

3.  **Install Dependencies.**
    ```bash
    pip install torch torchvision pytorch-lightning omegaconf solo-learn torchmetrics
    ```
    *Our `evaluation.py` script contains the `LinearModel` class but relies on `solo-learn` for some utility functions.*

4.  **Run Evaluation.** From the root directory of this project, run the evaluation script.

    *   **To reproduce the 4-Class MoCo-v3 result:**
    ```bash
    python evaluation/evaluation.py \
        --task 4class \
        --ckpt "./models/mocov3-brain4class.ckpt"
    ```
    *   **To reproduce the 16-Class SimCLR result:**
    ```bash
    python evaluation/evaluation.py \
        --task 16class \
        --ckpt "./models/simclr-brain16class.ckpt"
    ```
    *   **To run a custom 6-class task:**
    ```bash
    python evaluation/evaluation.py \
        --task custom \
        --ckpt "path/to/your/model.ckpt" \
        --custom_task_classes 6 \
        --custom_dataset_path "/path/to/your/dataset" \
        --backbone resnet18
    ```

***

# Notes
*   **Reproducibility**: Please be aware that due to differences in hardware, PyTorch, and cuDNN versions, your results may show minor variations (typically within ±0.5%) compared to the numbers reported in our manuscript.

*   **Training**: All models were trained using the [**solo-learn**](https://github.com/vturrisi/solo-learn) framework. Our evaluation script contains the minimal code necessary to validate the models and relies on `solo-learn` for certain utilities.

# Acknowledgements
We would like to thank the authors of `solo-learn` for providing a comprehensive and high-quality library for the self-supervised learning community. If you use our evaluation code, which leverages their framework, please also consider citing their paper.

```bibtex
@article{JMLR:v23:21-1155,
  author  = {Victor Guilherme Turrisi da Costa and Enrico Fini and Moin Nabi and Nicu Sebe and Elisa Ricci},
  title   = {solo-learn: A Library of Self-supervised Methods for Visual Representation Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {56},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v23/21-1155.html}
}
```
