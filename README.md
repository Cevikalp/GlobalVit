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
-- We demonstrate that the proposed architecture can effectively address
face frontalization by reconstructing structurally consistent frontal face images.
-- We show that the same framework can be extended to semantic segmentation with minimal architectural changes, illustrating its 
general applicability to dense vision tasks.
-- Extensive experiments on benchmark datasets validate the effectiveness and flexibility of the proposed method.

<img width="1254" height="350" alt="transformer_global_vit" src="https://github.com/user-attachments/assets/8a4b0e20-a85b-46d8-9101-9e5d7f9dffdf" />

**Fig 1.**  Illustration of the proposed Global Image Transformer (GIT) framework. Depending on the target
task, the model can take either non-frontal face images or RGB scene images as input and produce the
corresponding outputs, such as frontalized face images or semantic segmentation maps, as shown in the
figure. Task-specific loss functions are employed during training to optimize the network for the respective
objective.
