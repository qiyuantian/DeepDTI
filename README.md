# DeepDTI

## s_DeepDTI_prepData.m

MATLAB script for preparing the input and ground-truth data for convolutional neural network in DeepDTI.

**Utility functions**

- *amatrix.m*: create diffusion tensor transformation matrix for given b-vectors

- *bgr_colormap.m*: create blue-gray-red color map for visualizaing residual images

- *decompose_tensor.m*: decompose diffusion tensors and derive DTI metrics

- *rot3d.m*: create 3D rotation matrix to rotate b-vectors

## **HCP data**

The example data are provided by the WU-Minn-Oxford Hhuman Connectome Project (HCP) (open access data use term at https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms). Please acknowledge the source to the WU-Minn-Oxford HCP. The orginal data is available at https://www.humanconnectome.org/study/hcp-young-adult.

## **Refereces**

[1] Tian Q, Bilgic B, Fan Q, Liao C, Ngamsombat C, Hu Y, Witzel T, Setsompop K, Polimeni JR, Huang SY. [DeepDTI: High-fidelity Six-direction Diffusion Tensor Imaging using Deep Learning](https://www.sciencedirect.com/science/article/pii/S1053811920305036). *NeuroImage*, 2020; 219: 117017. [[**PDF**](https://reader.elsevier.com/reader/sd/pii/S1053811920305036?token=4F6CB875114A041666DB65DA13CC2051F735735B722175386824481FFCE217C9C352F6B55830BAF29A913441E4B5FF5B&originRegion=us-east-1&originCreation=20210627165514)]

[2] Tian Q, Li Z, Fan Q, Ngamsombat C, Hu Y, Liao C, Wang F, Setsompop K, Polimeni JR, Bilgic B, Huang SY. [SRDTI: Deep learning-based super-resolution for diffusion tensor MRI.](https://arxiv.org/abs/2102.09069) *arXiv Preprint*, 2021; arXiv: 2102.09069. [[**PDF**](https://arxiv.org/pdf/2102.09069.pdf)]
