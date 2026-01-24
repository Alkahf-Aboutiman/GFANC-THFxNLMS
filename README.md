# Hybrid Deep Learning–Based Active Noise Control (GFANC-THFxNLMS)

This repository provides the implementation of the hybrid active noise control (ANC) algorithm **GFANC-THFxNLMS**.

## Overview

Conventional ANC systems usually assume linear signal paths. However, in real applications, nonlinear distortions—especially those introduced by loudspeakers—can significantly degrade performance.

This work proposes a **hybrid deep learning and adaptive filtering framework** that improves robustness in nonlinear and time-varying acoustic environments, particularly for **encapsulated structures with openings**.

The system consists of two main stages:

- **GFANC (Deep Learning Stage)**  
  A dataset of sub-control filters is generated offline. A 1D convolutional neural network (CNN) generates the most suitable weights to design the control filter according to the incoming noise.

- **THFxNLMS (Adaptive Stage)**  
  The CNN-generated filter is further optimized online using an adaptive filtering algorithm to compensate for modeling errors and environmental variations.

By combining data-driven filter generation with real-time adaptation, the proposed method achieves improved noise attenuation compared to conventional approaches.

---

## Dataset

The dataset used in this work is publicly available on Zenodo:

🔗 https://doi.org/10.5281/zenodo.18361957

## Acknowledgements

This work is inspired by the following open-source repositories:

- https://github.com/Luo-Zhengding/SFANC-FxNLMS-ANC-Algorithm-based-on-Deep-Learning  
- https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control  

We sincerely thank the authors for making their code publicly available.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{aboutiman2026hybrid,
  title={Hybrid Deep Learning--based Active Noise Control for Encapsulated Structures with Openings},
  author={Aboutiman, Alkahf and Maamoun, Khaled Said Ahmed and Karimi, Hamid Reza and Ripamonti, Francesco},
  journal={Expert Systems with Applications},
  doi = {https://doi.org/10.1016/j.eswa.2026.131247},
  url = {https://www.sciencedirect.com/science/article/pii/S0003682X25002956},
  pages={131247},
  year={2026},
  publisher={Elsevier}
}


