# brain-MRI-SSL
This is the official repository for the paper 
Tailored self-supervised pretraining improves brain MRI diagnostic models,
by Xinhao Huang, Zihao Wang, Weichen Zhou, Kexin Yang, Kaihua Wen, Haiguang Liu, Shoujin Huang, Mengye Lyu
Computerized Medical Imaging and Graphics,Volume 123,2025,102560,ISSN 0895-6111,
https://doi.org/10.1016/j.compmedimag.2025.102560.
(https://www.sciencedirect.com/science/article/pii/S0895611125000692)
Abstract: Self-supervised learning has shown potential in enhancing deep learning methods, yet its application in brain magnetic resonance imaging (MRI) analysis remains underexplored. This study seeks to leverage large-scale, unlabeled public brain MRI datasets to improve the performance of deep learning models in various downstream tasks for the development of clinical decision support systems. To enhance training efficiency, data filtering methods based on image entropy and slice positions were developed, condensing a combined dataset of approximately 2 million images from fastMRI-brain, OASIS-3, IXI, and BraTS21 into a more focused set of 250 K images enriched with brain features. The Momentum Contrast (MoCo) v3 algorithm was then employed to learn these image features, resulting in robustly pretrained models specifically tailored to brain MRI. The pretrained models were subsequently evaluated in tumor classification, lesion detection, hippocampal segmentation, and image reconstruction tasks. The results demonstrate that our brain MRI-oriented pretraining outperformed both ImageNet pretraining and pretraining on larger multi-organ, multi-modality medical datasets, achieving a ∼2.8 % increase in 4-class tumor classification accuracy, a ∼0.9 % improvement in tumor detection mean average precision, a ∼3.6 % gain in adult hippocampal segmentation Dice score, and a ∼0.1 PSNR improvement in reconstruction at 2-fold acceleration. This study underscores the potential of self-supervised learning for brain MRI using large-scale, tailored datasets derived from public sources.
Keywords: Self-supervised learning; Representation learning, Feature extraction; Brain imaging, Tumor classification


