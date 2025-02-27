# Link Review

- here I have collect info about all the works that may be useful for writing this paper
- I divide these works by to topic in order to struct them

> [!Note]
> This review table will be updated, so it is not a final version

| Topic | Title | Year | Authors | Paper | Code | Summary |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| Main articles | U-Net: Convolutional Networks for Biomedical Image Segmentation | 2015 | Olaf Ronneberge et al. | [arXiv](https://arxiv.org/abs/1505.04597) | [link](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net) | TODO |
| Large T2I diffusion models | [DALL-E 2]  Hierarchical text-conditional image generation with clip latents | 2022 | Ramesh, A. et al. | [arXiv](https://arxiv.org/abs/2204.06125) | - | TODO. First try to work with image prompt. |
|      | [Stable Diffusion (SD)] High-Resolution Image Synthesis with Latent Diffusion Models| 2022 | Robin Rombach et al. | [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper) | [GitHub](https://github.com/CompVis/latent-diffusion) | TODO. Only text prompt. |
|      | Stable unCLIP | - | Robin Rombach et al. | based on [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper) | [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip) | TODO |
| Control tools and adapters | [ControlNet] Adding Conditional Control to Text-to-Image Diffusion Models | 2023 | Lvmin Zhang et al. | [ICCV Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html) | - | TODO | 
|      | T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models | 2024 | Mou, C. et al. | [AAAI Conference](https://ojs.aaai.org/index.php/AAAI/article/view/28226) | [GitHub](https://github.com/TencentARC/T2I-Adapter) | Simple style adapter. Image features extracted from CLIP image encoder are mapped to new features by a trainable network and then concatenated with text features. The merged features are fed into the UNet of the diffusion model to guide image generation. The results often worse than fine-tuned image prompt models. |
| ODE solvers for diffusion models| [DDIM] Denoising Diffusion Implicit Models | 2020 | Jiaming Song et al. | [arXiv](https://arxiv.org/abs/2010.02502) | - | TODO |
|      | [PNDM] Pseudo numerical methods for diffusion models on manifolds | 2022 | L Liu et al. | [arXiv](https://arxiv.org/abs/2202.09778) | - | TODO? |
|      | Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models | 2022 | C Lu et al. | [arXiv](https://arxiv.org/abs/2211.01095) | - | TODO? |
| Some models and info for me | OpenCLIP | 2022 | Ilharco Gabriel et al. | - | [GitHub](https://github.com/mlfoundations/open_clip) | TODO |

