# RescueNet Segmentation Project — Full Development History (BWSI July 2025 → Oct 2025)

This document outlines the entire development process of **RescueNet**, an aerial-image segmentation model designed to detect and classify post-disaster features such as damaged structures, flooded areas, roadways, and vehicles.
Originally developed as a classroom prototype at the MIT Beaver Works Summer Institute (BWSI), it underwent many  iterations from July to October 2025, transforming into a refined, research-grade system.

It would be recommended to read the documentation to learn about my journey with this model before diving into this version history.
Some terminology:
- **mIoU (Mean Intersection-over-Union):** Average overlap between predicted and ground-truth classes across all categories.  
- **OHEM (Online Hard Example Mining):** Loss term focusing on the most difficult pixels to classify correctly, improving rare-class learning.
---

## Overview

**Objective:** To categorize each pixel in satellite images taken after a disaster into 11 distinct semantic categories, such as water, roads, buildings (classified by damage level), vehicles, trees, and pools.
This procedure, referred to as **semantic segmentation**, transforms unprocessed aerial images into organized data for disaster mapping.

**Dataset:** RescueNet (≈22 GB ZIP, includes high-resolution aerial photos and corresponding labeled masks)

**Final Model Performance**
- Validation mIoU: **0.724** (≈72.4% mean overlap accuracy)  
- Pixel Accuracy: **≈90%**

**Core Architecture:** DeepLabV3+ with ResNet-50 backbone  
**Loss Functions:** Cross-Entropy, Lovasz, Focal CE, OHEM  
**Optimizer:** AdamW (learning rate 1e-4, weight decay 1e-4)  
**Scheduler:** Cosine learning rate with warm-up  
**Sampler:** Rarity-aware (prioritizes rare classes like vehicles or pools)  
**Precision:** AMP (mixed 16/32-bit)  
**EMA:** 0.999 decay for weight averaging  
**TTA:** Multi-scale [0.85, 1.0, 1.15] with horizontal flip averaging

---

## Version 0 — BWSI Prototype (July 2025)

**Environment:** Google Colab Free Tier (T4/P100 GPU, 12 GB VRAM)

The initial version was developed during the **MIT Beaver Works Summer Institute’s "Remote Sensing for Disaster Response"** course.  
It was **conceptually based on a UNet architecture with a ResNet-18 backbone**, serving as a lightweight proof of concept for semantic segmentation on disaster imagery.  
While the full UNet training wasn’t preserved in logs, several commented cells document this setup, forming the conceptual foundation for later DeepLabV3+ experiments.

**Configuration**
- Prototype UNet (ResNet-18 backbone, pretrained on ImageNet)  
- Basic augmentations (blur, sharpness, horizontal/vertical flips)  
- AdamW optimizer (lr 1e-4, wd 1e-4)  
- Batch size = 2, crop 256 × 256  

**Limitations**
- Unweighted loss ignored rare categories  
- No mixed precision or EMA  
- Small crops limited spatial context  
- Model sometimes misclassified small or complex regions  

**Result**
- Pixel Accuracy: ≈ 65 %  
- Loss ≈ 0.8  
- Detected large features (water, buildings) but failed on edges and vehicles  
- Unstable due to VRAM limits on free Colab  

--
## Version 1 — Dataset Pipeline Rebuild (Aug 10, 2025)

The dataset pipeline was rebuilt for reliability and reproducibility.  

**Key Updates**
- Automated Dropbox → Drive caching (22 GB)
- RGB→Class ID lookup table (LUT) for exact mask decoding
- `IGNORE_INDEX = 255` for invalid pixels
- Padding added for uniform image size

**Result:**  
A clean, reproducible dataset ready for large-scale training.

---

## Version 2 — First Stable Training (Aug 17 2025)

Training began with the architecture: **DeepLabV3+ (ResNet-50)**.  
Weighted losses, warm-up cosine scheduling, and mixed precision were introduced for the first time.  
This marks the first version with sustained multi-epoch training and measurable improvement in mIoU.

**Key Additions**
- Weighted Cross-Entropy + Lovasz + Tversky (α = 0.6, β = 0.4)  
- Cosine learning-rate schedule with 1-epoch warm-up  
- Mixed precision (fp16) and EMA (0.999)  
- Batch 16 (with gradient accumulation)  
- Crop 768 × 768  

**Result**
- Validation mIoU: ≈ 0.6 (Different than pixel accuracy or loss)
- Much steadier losses, though still biased toward frequent classes (buildings, water)


---

## Version 3 — Rarity-Aware Sampler and Hybrid Loss (Aug 21, 2025)

This version addressed class imbalance. A **rarity-aware sampler** ensured that rare categories (vehicles, pools, blocked roads) appeared more frequently in training.  
The **hybrid loss** combined four complementary objectives.

**Changes**
- Sampler formula: rarity = presence^1.1 × area^0.9, clamped [0.25, 6]
- Loss composition:
  - 0.35 Weighted Cross-Entropy  
  - 0.30 Lovasz  
  - 0.20 Focal CE  
  - 0.15 OHEM (top 20% hardest pixels)
- Gradient clipping at 1.0  
- AMP (bf16) with `channels_last` memory format  

**Result**
- Validation mIoU: 0.65
- Strong improvements for rare classes based on test images

---

## Version 4 — Augmentation and Normalization Refinement (Aug 30, 2025)

Transformations were standardized for consistent data input.  
Motion blur and color jitter were tuned to reflect real-world drone photography variability.

**Pipeline**
`LongestMaxSize → SmallestMaxSize → PadIfNeededConst → CropNonEmptyMaskIfExists → Normalize`

**Result:**  
- Cleaner boundaries, fewer data errors, and improved epoch-to-epoch stability.
- Validation mIoU: 0.7
---

## Version 5 — Evaluation and Visualization Suite (Sep 9, 2025)

An internal evaluation framework was developed to analyze and visualize model performance.  
This enabled faster debugging and deeper insight into error sources.

**Features**
- Per-class IoU and confusion matrix
- Test-time augmentation (TTA): scales [0.85, 1.0, 1.15], flip averaging
- Visual overlays combining predictions and ground truth
- Optional small-component cleanup  

**Result:**  
Metrics and visuals aligned, confirming model generalization.

---

## Version 6 — Resource Management and Stability (Late Sept 2025)

Focus shifted toward long-term stability and reproducibility.

**Enhancements**
- RAM/VRAM tracking with `psutil` and `torch.cuda.mem_get_info()`
- Safe model saving and loading
- Reduced log clutter and checkpoint conflicts  

**Result:**  
Training sessions became stable across runs; no more GPU crashes.

---

## Version 7 — Checkpoint Management (Oct 8, 2025)

Structured versioning and automatic backups were added.

**Improvements**
- Versioned filenames: `v13_deeplabv3p_ema_e{epoch}.pth`
- Automatic Drive sync for best checkpoints
- Timestamped logs  

**Result:**  
Clear rollback control and traceable experiment history.

---

## Version 8 — Safe Inference and Visualization (Oct 9, 2025)

Inference (testing) was separated from training to ensure consistent and reproducible results.

**Changes**
- Dedicated inference notebook cells
- Fixed validation transformations
- Optional small-object filtering  

**Result:**  
Deterministic outputs and polished visualization-ready overlays.

---

## Version 9 — Final Integration (Oct 10, 2025)

All improvements were merged into a single, unified training and inference pipeline.

**Configuration**
- DeepLabV3+ + ResNet-50 backbone  
- Rarity-aware sampler + hybrid loss  
- EMA and AMP for stability  
- TTA with automated logging and GPU tracking  

**Results**
- Validation mIoU: **0.724**  
- Pixel Accuracy: **≈90%**  
- Strongest classes: water and medium-damage buildings (~0.82 IoU)  
- Weakest classes: vehicles and blocked roads (~0.45–0.55 IoU)
> *Note:* Earlier experiments briefly explored a PSPNet (EfficientNet-B3) branch for comparison, but it was later retired in favor of the more stable DeepLabV3+ pipeline.

---

## Cumulative Lessons

- Data balance influenced results more than model architecture.  
- EMA and mixed precision stabilized training and improved reliability.  
- Combining Lovasz and Focal losses enhanced both edge quality and rare-class recognition.  
- Visualization tools provided intuitive feedback and sped up debugging.  
- Proper versioning and data consistency enabled full reproducibility.

---

**Summary:**  
Over the course of multiple iterations, RescueNet transformed from a delicate prototype into a strong and replicable disaster-response framework. Each modification - be it in data management, optimization, or assessment - advanced its relevance to practical applications.
