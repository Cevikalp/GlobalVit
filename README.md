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
- For face frontalization, simply run  **'main_ViT_Frontalization.py'**. This script initializes and trains a model from scratch with randomly initialized weights.
  The other two training scripts utilize pretrained transformer encoders designed for face analysis. Specifically, **'main_frontalization_pretrained_encoder.py'**  employs a pretrained
  [Transface](https://github.com/DanJun6737/TransFace?utm_source=chatgpt.com) encoder, while **'main_frontalization_pretrained_encoder_v2.py'** uses a [FaceXFormer](https://github.com/Kartik-3004/facexformer) encoder.

### Results
- To reproduce the Honda test results given in the journal version of the paper, just run **'main_test_honda_images.py'**. It will produce MSE and SSIM scores. Once you run this script, it will produce a directory with the frontalized images. Then, run
  **'compute_fid_from_frontal-images.py'** to produce FID scores. The test images are given under FaceDatasets directory.

![papernew](https://github.com/user-attachments/assets/0994eba8-38ef-43e6-8412-ae93831f3c43)

**Fig 2.** Visualization of frontalization results for selected non-frontal face images. The first row presents the input images with pose variations, while the second row shows the corresponding ground-truth frontal
references. The subsequent rows display the outputs produced by the evaluated methods.

- If you want to test the trained models on non-aligned face images, you can run the **'align_frontalize_faces.py'** script. It first applies face alignment 
using RetinaFace to normalize the images, and then performs frontalization on the aligned faces. We also collected a set of images from the internet and our lab environment, 
available in the real_images directory. To frontalize these images, run the **'test-real-images.py'** script. This script allows you to choose among four different 
face frontalization models, including variants based on a DETR-style decoder, a GIT decoder, and models that use pretrained face transformer encoders. The outputs of the tested
methods for these images can be seen below.


![real_tests](https://github.com/user-attachments/assets/56dd771b-a002-40b4-81b0-a6c0adf2cd23)

**Fig 3.** Visualization of frontalization results for images given under real_images directory.

## Semantic Segmentation
### Training
- We trained two different models on Ade20K and CityScapes datasets. Simply run, **'main_transformer_segmentation_Ade20K_Dino.py'** script to train on Ade20K dataset and run **'main_transformer_segmentation_Cityscapes_Dino.py'**
  script to train on cityScapes dataset.
### Results
- To reproduce the semantic segmentation results given in the paper, run the scripts **'evaluate_ade20k.py'** and **'evaluate_cityscapes.py.py'** given under evaluation_segmentation directory.

  
![semnatic_segmentation](https://github.com/user-attachments/assets/47ec226a-ee2f-4635-9f5e-449e48958590)

**Fig 4.** Visualization of the semantic segmentation results produced by the proposed method. The first row presents the input images, while the second row shows the corresponding ground-truth masks. The final
row illustrates the segmentation outputs generated by the proposed method.

## Citation
```bibtex
@inproceedings{cevikalpcvpr,
  author    = {Hakan Cevikalp and Hasan Saribas and Kaya Turgut},
  title     = {Frontal Face Synthesis by Using Vision Transformers},
  booktitle = {IEEE Society Conference on Computer Vision and Pattern Recognition (CVPR) Workshops },
  year      = {2026},
}

@article{cevikalp2026,
  author    = {Hakan Cevikalp and Hasan Saribas and Kaya Turgut and Faruk Dirisaglik},
  title     = {A Global Transformer Framework for Image-to-Image Translation},
  journal = {Pattern Recognition},
  year      = {under review},
}

# Contact
If you have any question about our work, please do not hesitate to contact us by email hakan.cevikalp@gmail.com.

