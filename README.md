# GlobalVit
This repo includes software for image-to-image translation by using a global transformer architecture.
# A Global Transformer Framework for Image-to-Image Translation

**Abstract:** In this paper, we propose a global transformer framework for image-to-image translation that leverages the self-attention mechanism to model global relationships between
image regions. The proposed architecture represents images as sequences of patch embeddings enriched with positional information and processes them using transformer
encoder–decoder modules to capture long-range contextual interactions. A lightweight convolutional refinement stage is employed to restore spatial details and produce highquality pixel-level outputs. The proposed framework is task-agnostic and can be applied to a variety of dense vision problems without significant architectural modifications. We demonstrate the effectiveness of the approach on two representative challenging image-to-image translation tasks: face frontalization and semantic segmentation. Extensive experiments on benchmark datasets show that the proposed method produces structurally consistent frontal face images and achieves competitive segmentation performance compared with existing methods. These results highlight the potential of transformer-based architectures for modeling global image structure and solving
diverse image-to-image translation problems.

**Our main contributions are as follows:**

-- We propose a global transformer-based framework for image-to-image
translation that leverages self-attention to model long-range spatial dependencies.

-- We demonstrate that the proposed architecture can effectively address
face frontalization by reconstructing structurally consistent frontal face images. 

-- We show that the same framework can be extended to semantic segmentation with minimal architectural changes, illustrating its 
general applicability to dense vision tasks. 

-- Extensive experiments on benchmark datasets validate the effectiveness and flexibility of the proposed method.

<img width="1254" height="350" alt="transformer_global_vit" src="https://github.com/user-attachments/assets/8a4b0e20-a85b-46d8-9101-9e5d7f9dffdf" />

**Fig 1.**  Illustration of the proposed Global Image Transformer (GIT) framework. Depending on the target
task, the model can take either non-frontal face images or RGB scene images as input and produce the
corresponding outputs, such as frontalized face images or semantic segmentation maps, as shown in the
figure. Task-specific loss functions are employed during training to optimize the network for the respective
objective.

# 1. Requirements
## Environments
Following packages are required for this repo.

    - python 3.10.18+
    - torch  2.4+
    - torchvision 0.19+ 
    - torch 1.9+
    - CUDA 12.9+
    - cython 3.1.4+
    - scikit-learn 1.3+
    - numpy 2.2.6+
    - tqdm 4.67.1
    - bcolz 1.2.1  
    - matplotlib 3.7.5
    - albumentations
    - blas 1.0+
    - insightface 0.7.3+
    - ninja 1.13+
    - opencv-contrib-python 4.10.0.84
    - pillow 11.3+
    - pytorch-cuda 12.4
    - pycocotools 2.0.8+
    - scipy 1.15.3+
    - scikit-learn 1.7.2+
    - scikit-image 0.25.2+
    - seaborn 0.13.2
    - torchvision 0.25.0.dev20251222+cu128
    - transformers 5.3+  

# 2. Training & Evaluation
## Frontalization
### Training
- For face frontalization, just run  **'main_ViT_Frontalization.py'**. It creates a model by starting completely from random values. The other two training codes use pre-trained encoders of transformes used for face analysis. 
  **'main_frontalization_pretrained_encoder.py'** uses petrained [Transface](https://github.com/DanJun6737/TransFace?utm_source=chatgpt.com) encoder whereas **'main_frontalization_pretrained_encoder_v2.py'** uses
  [FaceXFormer](https://github.com/Kartik-3004/facexformer) encoder.

### Results
- To reproduce the Honda test results given in the paper, just run **'main_test_honda_images.py'**. It will produce MSE and SSIM scores. Once you run this script, it will produce a directory with the frontalized images. Then, run
  **'compute_fid_from_frontal-images.py'** to produce FID scores. The test images are given under FaceDatasets directory.

![paper_fig](https://github.com/user-attachments/assets/33da9cc5-fc25-4a56-98e5-8ec1fd35310a)

**Fig 2.** Visualization of frontalization results for selected non-frontal face images. The first row presents the input images with pose variations, while the second row shows the corresponding ground-truth frontal
references. The subsequent rows display the outputs produced by the evaluated methods.
