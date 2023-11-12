# Customizing Multimodal Foundation Models

This folder contains multiple examples for techniques that can be used to customize multimodal models, specifically focusing on Stable Diffusion. 

High level description of included examples: 

* [01_controlnet.ipynb](./01_controlnet.ipynb): This example shows how you can use cutting edge techniques to augment diffusion to better handle common image tasks such as edge detection and segmentation maps. These techniques provide fine-grained control over image generation.

* [02_textual_inversion.ipynb](./02_textual_inversion.ipynb): This example shows a technique called textual inversion that allows you to customize a Stable Diffusion model by creating a learnable token representing your customized subject while keeping the base model remains frozen.

* [03_dreambooth.ipynb](./03_dreambooth.ipynb): This example shows a full fine-tuning technique that uses a method called DreamBooth.

* [04_dreambooth_LoRA.ipynb](./04_dreambooth_LoRA.ipynb): This example demonstrates a fine-tuning technique for Stable Diffusion that focuses on LoRA based fine-tuning of the cross-attention layers. 

* [05_rlhf_ddpo_sd.ipynb](./05_rlhf_ddpo_sd.ipynb): This example shows an implementation of Denoising Diffusion Policy Optimization (DDPO) in PyTorch with support for low-rank adaptation (LoRA) for fine-tuning Stable Diffusion. 