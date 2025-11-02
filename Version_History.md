# RescueNet Segmentation Project — Full Development History (BWSI → Oct 2025)

This document presents the full evolution of **RescueNet**, an aerial-imagery segmentation model designed to detect and classify post-disaster features such as damaged buildings, flooded areas, roads, and vehicles.  
The project began as a classroom prototype during the **MIT Beaver Works Summer Institute (BWSI)** and developed into a mature, research-grade system over ten iterative versions between July and October 2025.

---

## Understanding the Foundations

Before diving into the version history, it may be useful to understand the principles behind how the model learns.  
RescueNet, like most modern computer vision systems, is built on deep learning — a form of machine learning where a neural network discovers patterns from data rather than following explicit instructions.

A **neural network** consists of interconnected mathematical “neurons.” Each neuron processes small fragments of information, and collectively they can recognize complex visual structures. During **training**, the model is shown many satellite images paired with “masks” that mark what each pixel represents — water, building, road, and so on. Over time, it adjusts millions of internal parameters to minimize mistakes.

The level of error is measured by a **loss function**. Think of it as a numerical score showing how far the model’s guesses are from the correct answers. A lower loss means better predictions. Several specialized losses are used in this project:
- **Cross-Entropy (CE):** measures general classification accuracy.
- **Lovasz Loss:** sharpens object boundaries for cleaner segmentation.
- **Focal Loss:** emphasizes rare or difficult examples so the model doesn’t ignore them.
- **OHEM (Online Hard Example Mining):** selects the hardest pixels for the model to learn from.

To reduce this loss, an **optimizer** such as **AdamW** fine-tunes the network’s internal weights after each iteration. The **learning rate** determines how aggressively these updates happen — too high, and learning becomes unstable; too low, and it stagnates.  
A **scheduler** dynamically adjusts this rate during training; here, a *cosine schedule* gradually lowers it for smoother convergence.

The model architecture combines two key components:
- **ResNet-50 Backbone:** A pre-trained feature extractor that identifies textures, edges, and shapes from input images.  
- **DeepLabV3+:** A segmentation network that classifies each pixel using context from nearby regions, producing a color-coded map of the scene.

Additional elements such as **automatic mixed precision (AMP)**, which uses a mix of 16-bit and 32-bit operations to save memory, and **EMA (Exponential Moving Average)**, which smooths model updates, improve both performance and stability.

These fundamentals — data, architecture, loss, and optimization — shaped the evolution of RescueNet across all ten versions documented below.

---

## Overview

**Objective:** Classify every pixel in post-disaster satellite imagery into 11 semantic categories, including water, roads, buildings (by damage level), vehicles, trees, and pools.  
This process, known as **semantic segmentation**, converts raw aerial images into structured disaster-mapping data.

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

The first version was built during the Beaver Works Summer Institute “Remote Sensing for Disaster Response” course.  
It used **DeepLabV3+** with a **ResNet-50** backbone pretrained on ImageNet and was trained using basic augmentations and a single Cross-Entropy loss.

**Configuration**
- Dataset manually unzipped to `/content/`
- DeepLabV3+ (ResNet-50 backbone)
- Basic augmentations (horizontal/vertical flips, color jitter)
- Cross-Entropy loss (no weighting)
- AdamW optimizer (lr 1e-4, wd 1e-4)
- Batch 4, crop 512×512  

**Limitations**
- No RGB-to-class-ID mapping caused mask mismatch noise  
- Unweighted loss ignored rare categories  
- No mixed precision or EMA  
- Small crops limited spatial context  

**Result**
- Validation mIoU: 0.52–0.55  
- Pixel Accuracy: ≈83%  
- Model recognized only large features (water, buildings)  
- Unstable training due to limited GPU memory  

---

## Version 1 — Dataset Pipeline Rebuild (Aug 10, 2025)

The dataset pipeline was rebuilt for reliability and reproducibility.  
Images and masks were now paired automatically and converted into consistent class IDs using a color lookup table.

**Key Updates**
- Automated Dropbox → Drive caching (22 GB)
- RGB→Class ID lookup table (LUT) for exact mask decoding
- `IGNORE_INDEX = 255` for invalid pixels
- Padding added for uniform image size
- Filename-based pairing with error handling  

**Result:**  
A clean, reproducible dataset ready for large-scale training.

---

## Version 2 — First Stable Training (Aug 17, 2025)

Training began with two architectures: DeepLabV3+ (ResNet-50) and PSPNet (EfficientNet-B3).  
Weighted losses and learning rate scheduling were introduced for the first time.

**Key Additions**
- Weighted CE + Lovasz + Tversky (α=0.6, β=0.4)
- Cosine learning rate schedule with 1-epoch warm-up
- Mixed precision (fp16) and EMA (0.999)
- Batch size 32 (with gradient accumulation)
- Crop size 768×768  

**Result**
- Validation mIoU: 0.68–0.70  
- More stable and consistent training, though still biased toward frequent classes  

---

## Version 3 — Rarity-Aware Sampler and Hybrid Loss (Aug 24, 2025)

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
- Validation mIoU: 0.714  
- Pixel Accuracy: ≈89%  
- Strong improvements for rare classes (+10–12 IoU points)

---

## Version 4 — Augmentation and Normalization Refinement (Aug 30, 2025)

Transformations were standardized for consistent data input.  
Motion blur and color jitter were tuned to reflect real-world drone photography variability.

**Pipeline**
`LongestMaxSize → SmallestMaxSize → PadIfNeededConst → CropNonEmptyMaskIfExists → Normalize`

**Result:**  
Cleaner boundaries, fewer data errors, and improved epoch-to-epoch stability.

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

## Version 7 — PSPNet Branch Fine-Tuning (Early Oct 2025)

The PSPNet branch was fine-tuned for texture precision.  
Although visually smoother, it underperformed on edges.

**Configuration**
- Loss: 0.6 CE + 0.4 Tversky  
- Scheduler: ReduceLROnPlateau  
- Batch 16, crop 1024×1024  

**Result:**  
Validation mIoU ≈ 0.67; less precise edges, retained for ensemble potential.

---

## Version 8 — Checkpoint Management (Oct 8, 2025)

Structured versioning and automatic backups were added.

**Improvements**
- Versioned filenames: `v13_deeplabv3p_ema_e{epoch}.pth`
- Automatic Drive sync for best checkpoints
- Timestamped logs  

**Result:**  
Clear rollback control and traceable experiment history.

---

## Version 9 — Safe Inference and Visualization (Oct 9, 2025)

Inference (testing) was separated from training to ensure consistent and reproducible results.

**Changes**
- Dedicated inference notebook cells
- Fixed validation transformations
- Optional small-object filtering  

**Result:**  
Deterministic outputs and polished visualization-ready overlays.

---

## Version 10 — Final Integration (Oct 10, 2025)

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

---

## Cumulative Lessons

- Data balance influenced results more than model architecture.  
- EMA and mixed precision stabilized training and improved reliability.  
- Combining Lovasz and Focal losses enhanced both edge quality and rare-class recognition.  
- Visualization tools provided intuitive feedback and sped up debugging.  
- Proper versioning and data consistency enabled full reproducibility.

---

## Future Work

- Evaluate **SegFormer** encoders for faster inference.  
- Explore **semi-supervised learning** on unlabeled aerial data.  
- Apply **model distillation** to deploy lightweight versions.  
- Integrate **boundary-aware losses** (SoftDice, Boundary IoU) for sharper segmentation.  
- Transition to **PyTorch Lightning** for multi-GPU scalability.

---

**Summary:**  
Through ten iterations, RescueNet evolved from a fragile prototype into a robust and reproducible disaster-response model. Each revision—whether improving data handling, optimization, or evaluation—brought it closer to real-world applicability, demonstrating how careful iteration and disciplined engineering can transform a student project into a professional-grade system.
