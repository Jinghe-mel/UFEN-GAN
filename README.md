# UFEN-GAN

Code and models for **"Knowledge Distillation for Underwater Feature Extraction and Matching via GAN-synthesized Images"**.

This project introduces an **adaptive GAN-based image synthesis** method that estimates environment-specific underwater conditions, considering both noise distribution and forward scattering.

The paper is available on [IEEE](https://ieeexplore.ieee.org/document/11081457) and [arXiv](https://arxiv.org/abs/2504.08253).

### 🔑 Key Insight

>*Noise from randomly flowing particles and forward scattering are the core challenges affecting underwater feature matching.*

## 📷 Visual Example

<p align="center">
  <img src="example.png" alt="Demo Example" width="1000"/>
</p>

<p align="center">
  <strong>Figure:</strong> Example of synthetic images showing: (1) in-air image, (2) simplified model output, (3) with forward scattering, (4) with generated noise, and (5) target environment.
</p>



---

## 🔧 Inference Demo

A fast implementation for synthetic image generation is available in the `inference/` folder.

To run the demo:

```bash
cd inference
python synthetic_demo.py
```

Pretrained weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1PBefKnEHxgE8K_xqZ6prpdIxbJKfbE47?usp=drive_link)

Please place the weights into the `inference/weights/` folder.

---

## 🧪 Training GAN synthesis model

Run the training script: `inference/synthetic_training.py`; the total training will take around 2 hours.

```bash
cd inference
python synthetic_training.py --turbid_data /path/to/underwater --clear_data /path/to/clear_data
```

You need to prepare:
- A folder containing the target underwater environment images (set via `--turbid_data`)
- A folder of **paired in-air RGB and depth images** (set via `--clear_data`)

The folder structure should be:
```
clear_data/
├── images/
└── depths/
```

In our paper, we used the [NYU-v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) as the source of in-air RGB-D images.


## ✅ To-Do
- [x] Release GAN inference code
- [x] Release GAN training code
- [ ] (Optional) Release knowledge distillation training code

---

## 📚 Citation

If you use this code, please cite our paper:

```bibtex
@article{yang2025ufen_gan,
  author={Yang, Jinghe and Gong, Mingming and Pu, Ye},
  journal={IEEE Robotics and Automation Letters},
  title={Knowledge Distillation for Underwater Feature Extraction and Matching via GAN-Synthesized Images},
  year={2025},
  volume={10},
  number={9},
  pages={8866--8873},
  doi={10.1109/LRA.2025.3589805}
}
```

---
## 📌 Reference

Our method builds on parts of:  
https://github.com/astra-vision/GuidedDisent  
https://github.com/NVlabs/MUNIT
