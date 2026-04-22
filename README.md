# Multi-Resolution Facial Expression Paired Dataset

A multi-resolution, multi-expression paired face dataset for fine-tuning Flux-based image generation models. This dataset provides same-person, different-expression image pairs at various resolutions, designed to preserve the multi-resolution generation capabilities of pre-trained Flux models during fine-tuning.

## Table of Contents

- [Motivation](#motivation)
- [Source Datasets](#source-datasets)
- [Processing Pipeline Overview](#processing-pipeline-overview)
- [Step 1: Source Data Curation](#step-1-source-data-curation)
- [Step 2: Multi-Resolution Bucket Design](#step-2-multi-resolution-bucket-design)
- [Step 3: Bucket Allocation](#step-3-bucket-allocation)
- [Step 4: Super-Resolution](#step-4-super-resolution)
- [Step 5: Resize to Target Resolution](#step-5-resize-to-target-resolution)
- [Step 6: RAF Expression Generation via API](#step-6-raf-expression-generation-via-api)
- [Step 7: API Output Resize](#step-7-api-output-resize)
- [Dataset Statistics](#dataset-statistics)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [License](#license)
- [Citation](#citation)

---

## Motivation

Our goal is to fine-tune a **Flux Kontext** diffusion model for controllable facial expression generation using 3D Morphable Model (3DMM) representations. The core pipeline requires **same-person, different-expression image pairs** as training data.

A critical challenge in fine-tuning large pre-trained models like Flux is **catastrophic forgetting** — if the fine-tuning data contains only a single resolution (e.g., 512×512), the model loses its ability to generate high-quality images at other resolutions. Since Flux was pre-trained on data spanning a wide range of resolutions and aspect ratios, **we must mirror this multi-resolution distribution in our fine-tuning data** to preserve the base model's versatile generation capabilities.

This motivates the construction of a multi-resolution dataset where images are organized into resolution **buckets** that align with Flux's pre-trained resolution vocabulary.

---

## Source Datasets

We leverage three publicly available face datasets, each contributing different strengths:

| Dataset | Type | People | Expressions | Resolution | Characteristics |
|---------|------|--------|-------------|------------|-----------------|
| **Multi-PIE** | Paired | 249 | 2 (neutral + smile) | 128×128 | Large subject pool, low resolution, controlled lab conditions |
| **KDEF** | Paired | 70 | 7 (neutral + 6 basic) | 562×762 | Medium resolution, full expression set, frontal-pose selected |
| **Oulu-CASIA** | Paired | 80 | 7 (neutral + 6 basic) | 320×240 | Low-medium resolution, strong illumination variations |
| **RAF-DB** | Unpaired | 3,204 | 1 (neutral only, extracted) | Varied (91–1,200px) | Large-scale, diverse in-the-wild faces, single-label annotations |

> **Key distinction**: Multi-PIE, KDEF, and Oulu-CASIA provide naturally paired expressions (the same person photographed in multiple expressions). RAF-DB only provides single images with expression labels — it does **not** contain paired data. We extract neutral-expression images from RAF-DB and synthesize expression variants via a generative API (see [Step 6](#step-6-raf-expression-generation-via-api)).

---

## Processing Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Paired Datasets (v1)                         │
│              Multi-PIE / KDEF / Oulu-CASIA                      │
│                                                                 │
│  ① Curate (frontal, good lighting) ──► ② Design Buckets        │
│                                            │                    │
│  ③ Allocate to Buckets ──► ④ SeedVR2 Super-Resolution           │
│                                 │                               │
│                        ⑤ Resize to exact bucket size            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Unpaired Dataset (RAF)                        │
│                       RAF-DB                                    │
│                                                                 │
│  ① Extract neutral images ──► ② Allocate to Buckets            │
│                                     │                           │
│  ③ SeedVR2 Super-Resolution ──► ④ Resize to bucket size        │
│         │                                                       │
│  ⑤ Compute 2K-proportional resolution for API ──► ⑥ API Gen    │
│         │                                                       │
│  ⑦ Resize API output to exact bucket size                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Source Data Curation

### 1.1 Paired Datasets (Multi-PIE, KDEF, Oulu-CASIA)

From each dataset, we select only images meeting these criteria:

- **Frontal pose**: Near-zero yaw/pitch angle to ensure face visibility
- **Good illumination**: Frontal or evenly distributed lighting to avoid shadow artifacts
- **No occlusion**: No sunglasses, hands covering face, etc.

For **KDEF**, images were originally captured under multiple angles — we selected only the straight-ahead (`S` suffix) photographs. For **Multi-PIE**, we selected images from the frontal camera (camera 05_1) under neutral illumination conditions.

The curated data is organized as:

```
final_data_v1/
├── neutral/
│   ├── multi_pie_001_neutral.jpg
│   ├── kdef_AF01.jpg
│   └── oulu_001_neutral.jpg
├── happy/
├── sad/
├── angry/
├── surprise/
├── fear/
└── disgust/
```

### 1.2 RAF-DB: Neutral Image Extraction

RAF-DB provides single images with discrete expression labels (1=surprise, 2=fear, 3=disgust, 4=happy, 5=sad, 6=angry, 7=neutral). Since RAF-DB is an **unpaired** dataset — each image shows a different person in a single expression — it cannot directly provide expression pairs.

Our strategy: **extract all neutral-labeled images** (label=7) as seed images, then synthesize the remaining 6 expression variants using a generative API (see [Step 6](#step-6-raf-expression-generation-via-api)).

```
final_data_raf/
└── neutral/
    ├── raf_test_0001.jpg
    ├── raf_test_0002.jpg
    └── ...  (3,204 images)
```

---

## Step 2: Multi-Resolution Bucket Design

### 2.1 Why Multi-Resolution?

The Flux model family uses a **Diffusion Transformer (DiT)** architecture with **Rotary Position Embedding (RoPE)** for spatial position encoding. Unlike fixed-resolution models, Flux processes images as variable-length token sequences:

```
Image (H×W) → VAE Encode (8× downsample) → Patchify (2×2) → Token Sequence (L = H×W / 256)
```

During pre-training, Flux was exposed to images at many different resolutions and aspect ratios, allowing RoPE to learn position encodings across diverse spatial configurations. If we fine-tune only on a single resolution, the model will "forget" how to generate at other resolutions — a form of **catastrophic forgetting**.

> **Principle**: Fine-tuning data should cover the same resolution distribution as pre-training data.

### 2.2 Flux Official Resolution Buckets

From the Flux official source code (`src/flux/util.py`), the 17 preferred aspect ratios at ~1 megapixel are defined as:

```python
PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392),
    (800, 1328), (832, 1248), (880, 1184), (944, 1104),
    (1024, 1024),
    (1104, 944), (1184, 880), (1248, 832), (1328, 800),
    (1392, 752), (1456, 720), (1504, 688), (1568, 672),
]
```

These 17 aspect ratios range from ~0.43:1 (extreme portrait) to ~2.33:1 (extreme landscape), all at approximately 1,048,576 pixels (~1024²).

### 2.3 Multi-Tier Bucket Generation

To cover the full resolution range that Flux supports (256px to 2048px), we scale the 17 official aspect ratios to **10 pixel-area tiers**:

$$\text{Target Pixels}_{tier} = \text{base}^2, \quad \text{base} \in \{256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048\}$$

For each tier, each of the 17 aspect ratios is scaled:

$$\text{scale} = \sqrt{\frac{\text{Target Pixels}_{tier}}{W_{ref} \times H_{ref}}}$$

$$W_{new} = \text{round}\left(\frac{W_{ref} \times \text{scale}}{16}\right) \times 16, \quad H_{new} = \text{round}\left(\frac{H_{ref} \times \text{scale}}{16}\right) \times 16$$

where $(W_{ref}, H_{ref})$ is one of the 17 reference resolutions and the rounding to multiples of 16 satisfies the Flux architecture constraints:

| Component | Alignment Requirement | Reason |
|-----------|----------------------|--------|
| VAE Encoder | Multiple of 8 | 8× spatial downsampling |
| Patchify | Multiple of 16 | 2×2 patches on latent (8×2=16 in pixel space) |
| Sequence Packing | (W/8) and (H/8) must be even | Token packing requires even latent dimensions |

All three constraints are satisfied when W and H are multiples of **16**.

### 2.4 Face-Friendly Bucket Selection

Not all 17 aspect ratios are suitable for face images. Extreme landscape ratios (e.g., 1568×672, ~2.33:1) would require heavy cropping or distortion of face images. We select **7 face-friendly aspect ratios** from the 17:

| Aspect Ratio (at 1024 tier) | W:H Ratio | Orientation |
|------------------------------|-----------|-------------|
| 832 × 1248 | ~2:3 | Portrait |
| 880 × 1184 | ~3:4 | Portrait |
| 944 × 1104 | ~6:7 | Near-square (portrait) |
| 1024 × 1024 | 1:1 | Square |
| 1104 × 944 | ~7:6 | Near-square (landscape) |
| 1184 × 880 | ~4:3 | Landscape |
| 1248 × 832 | ~3:2 | Landscape |

With 7 aspect ratios × 10 tiers, filtering out buckets where any dimension exceeds 2048 pixels, we obtain **64 unique resolution buckets**.

### 2.5 Verification

All 64 buckets were verified to satisfy Flux's architecture constraints:

```python
for (W, H) in all_buckets:
    assert W % 16 == 0       # Patchify constraint
    assert H % 16 == 0
    assert (W // 8) % 2 == 0  # Packing constraint
    assert (H // 8) % 2 == 0
    assert W <= 2048 and H <= 2048  # Maximum dimension
```

---

## Step 3: Bucket Allocation

### 3.1 Allocation Strategy

Each person's images must be assigned to **exactly one bucket** (all expressions of the same person share the same resolution). The goals are:

1. **Uniform distribution**: Each bucket should contain approximately the same number of people
2. **Minimal distortion**: Assign each person to the bucket closest to their original resolution

### 3.2 Two-Phase Allocation

**Phase 1 — Initial Assignment**: Each person is assigned to the nearest bucket based on their original image resolution:

$$\text{bucket}^* = \arg\min_{(W_b, H_b)} \left| \frac{W_b}{H_b} - \frac{W_{orig}}{H_{orig}} \right|$$

With ties broken by pixel area proximity:

$$\text{(tie-break)} = \arg\min_{(W_b, H_b)} \left| W_b \cdot H_b - W_{orig} \cdot H_{orig} \right|$$

**Phase 2 — Rebalancing**: People are iteratively moved from the most populated bucket to the least populated bucket until all buckets differ by at most 1 person:

```
while max(bucket_counts) - min(bucket_counts) > 1:
    move one person from argmax(bucket_counts) to argmin(bucket_counts)
```

### 3.3 Allocation Results

| Dataset | People | Buckets | People per Bucket |
|---------|--------|---------|-------------------|
| **v1** (Multi-PIE + KDEF + Oulu) | 399 | 64 | 6–7 |
| **RAF** (neutral only) | 3,204 | 64 | 50–51 |

---

## Step 4: Super-Resolution

### 4.1 Why Super-Resolution?

Most source images are significantly smaller than their target bucket resolution:

| Source | Original Resolution | Target Range | Upscale Factor |
|--------|-------------------|--------------|----------------|
| Multi-PIE | 128×128 | 208×320 to 2048×2048 | 1.6× to 16× |
| Oulu-CASIA | 320×240 | 208×320 to 2048×2048 | 1× to 8.5× |
| KDEF | 562×762 | 208×320 to 2048×2048 | 0.4× to 3.6× |
| RAF-DB | 91–1200px (varied) | 208×320 to 2048×2048 | varies |

Simple interpolation (e.g., bicubic, Lanczos) produces blurry results at high upscale factors. We use **SeedVR2-7B**, a state-of-the-art one-step diffusion transformer model for image restoration, to perform high-quality face super-resolution.

### 4.2 SeedVR2-7B

SeedVR2 (ICLR 2026) is a one-step diffusion-based video/image restoration model that:

- Supports **arbitrary input and output resolutions** without patch-based processing
- Uses **adaptive window attention** for resolution-aware processing
- Achieves state-of-the-art perceptual quality in a **single inference step**
- Preserves facial identity through strong generative priors

The super-resolution is performed per-bucket: all images assigned to a given bucket are super-resolved to that bucket's target resolution in a single batch:

```bash
torchrun --nproc-per-node=8 inference_seedvr2_7b_img.py \
    --video_path INPUT_DIR \
    --output_dir OUTPUT_DIR \
    --res_h TARGET_H \
    --res_w TARGET_W \
    --sp_size 1
```

SeedVR2 internally applies `NaResize` to match the target pixel area while preserving aspect ratio, followed by `DivisibleCrop` for 16-pixel alignment.

---

## Step 5: Resize to Target Resolution

### 5.1 Why Resize After Super-Resolution?

SeedVR2's output resolution may not exactly match the target bucket dimensions due to its internal `NaResize` + `DivisibleCrop` pipeline, which:

1. Scales the input to match the **target pixel area** $\sqrt{H_{target} \times W_{target}}$
2. Crops to the nearest multiple of 16

This can produce outputs that are close to but not exactly the bucket's target $(W, H)$. For example, targeting 832×1248 might yield 832×1248 or 848×1232 depending on the input aspect ratio.

### 5.2 Final Lanczos Resize

After super-resolution, each image is resized to the **exact** bucket dimensions using Lanczos interpolation:

```python
img = img.resize((target_w, target_h), Image.LANCZOS)
```

Since SeedVR2 already produces a high-quality, high-resolution image, this final resize is typically a very minor adjustment (< 5% in either dimension) and introduces negligible quality loss.

---

## Step 6: RAF Expression Generation via API

### 6.1 Problem

RAF-DB contains only **single images per person** with expression labels. To create paired training data (same person, different expressions), we need to **synthesize 6 additional expressions** for each neutral image.

### 6.2 Solution: Generative API

We use the **doubao-seedream** image editing API to generate expression variants from each neutral face. For each neutral image, 6 expression-specific prompts guide the model to produce:

- Happy, Sad, Angry, Surprise, Fear, Disgust

The API preserves the person's identity while modifying only the facial expression, leveraging instruction-based image editing capabilities.

### 6.3 Resolution Handling for API Generation

The API has specific constraints:

- **Minimum pixel count**: The API requires input images above a minimum resolution threshold
- **Aspect ratio preservation**: Output images maintain the input's aspect ratio
- **Maximum output**: Supports up to ~2K resolution

Since our target bucket resolutions span from 208×320 to 2048×2048, directly requesting the API to generate at small bucket sizes (e.g., 208×320) would produce low-quality results. Instead, we adopt a **generate-high-then-downscale** strategy:

**Step 1: Compute the 2K-proportional resolution**

For a target bucket with dimensions $(W_b, H_b)$, we compute a proportionally scaled version where the pixel area matches ~2K (2048×2048 = 4,194,304 pixels):

$$\text{scale}_{2K} = \sqrt{\frac{2048^2}{W_b \times H_b}}$$

$$W_{api} = \text{round}\left(\frac{W_b \times \text{scale}_{2K}}{16}\right) \times 16, \quad H_{api} = \text{round}\left(\frac{H_b \times \text{scale}_{2K}}{16}\right) \times 16$$

This ensures:
- The API always generates at high resolution (near 2K) for maximum quality
- The aspect ratio is preserved: $\frac{W_{api}}{H_{api}} \approx \frac{W_b}{H_b}$
- The dimensions are aligned to 16-pixel multiples

**Example**: For target bucket 416×624 (~512 tier):

$$\text{scale}_{2K} = \sqrt{\frac{4,194,304}{416 \times 624}} = \sqrt{\frac{4,194,304}{259,584}} \approx 4.02$$

$$W_{api} = \text{round}(416 \times 4.02 / 16) \times 16 = 1680, \quad H_{api} = \text{round}(624 \times 4.02 / 16) \times 16 = 2512$$

**Step 2: Generate at 2K resolution**

```python
response = client.images.edit(
    model="doubao-seedream-5-0-260128",
    image=base64_neutral_image,
    prompt=expression_prompt,
    size=f"{W_api}x{H_api}",
    response_format="b64_json",
)
```

**Step 3: Resize to target bucket**

The high-resolution API output is then resized to the exact bucket dimensions:

```python
generated_img = generated_img.resize((W_b, H_b), Image.LANCZOS)
```

This two-step approach ensures that expression details (subtle muscle movements, wrinkles, teeth visibility) are generated at full quality before being downscaled, rather than being generated at a low resolution where these details would be lost.

---

## Step 7: API Output Resize

After API generation, all expression variants are resized to their exact bucket dimensions using high-quality Lanczos interpolation. This final step guarantees:

1. **Pixel-exact bucket alignment**: Every image in a bucket has exactly $(W_b, H_b)$ pixels
2. **Same-person resolution consistency**: All 7 images (1 neutral + 6 generated expressions) of the same person are at identical resolution
3. **Flux architecture compatibility**: All dimensions satisfy the 16-pixel alignment constraint

---

## Dataset Statistics

### Overall

| Metric | Value |
|--------|-------|
| Total people (v1 paired) | 399 |
| Total people (RAF synthesized) | 3,204 |
| Total people | 3,603 |
| Expressions per person | 7 (neutral + 6 basic) |
| Resolution buckets | 64 |
| Aspect ratios | 7 (face-friendly subset) |
| Pixel-area tiers | 10 (256² to 2048²) |
| Total images | ~25,221 |

### Per-Source Distribution

| Source | People | Expressions/Person | Original Resolution |
|--------|--------|--------------------|---------------------|
| Multi-PIE | 249 | 2 (neutral + smile) | 128×128 |
| KDEF | 70 | 7 | 562×762 |
| Oulu-CASIA | 80 | 7 | 320×240 |
| RAF-DB | 3,204 | 7 (1 real + 6 synthesized) | Varied |

### Bucket Distribution

People are uniformly distributed across all 64 buckets:

- **v1_bucket**: 6–7 people per bucket (399 total)
- **raf_bucket**: 50–51 people per bucket (3,204 total)
- **Combined**: 56–58 people per bucket

### Resolution Tiers

| Tier (base²) | Pixel Area | # Buckets | Aspect Ratios |
|--------------|-----------|-----------|---------------|
| 256² | ~65K | 7 | All 7 face-friendly |
| 384² | ~147K | 7 | All 7 |
| 512² | ~262K | 7 | All 7 |
| 640² | ~410K | 7 | All 7 |
| 768² | ~590K | 7 | All 7 |
| 896² | ~803K | 7 | All 7 |
| 1024² | ~1.05M | 7 | All 7 |
| 1280² | ~1.64M | 7 | All 7 |
| 1536² | ~2.36M | 5 | Filtered (max dim 2048) |
| 2048² | ~4.19M | 1 | Square only |

---

## Directory Structure

```
dataset/
├── final_data_v1_bucket/           # Paired datasets (Multi-PIE, KDEF, Oulu)
│   ├── 256x256/
│   │   ├── neutral/
│   │   │   ├── multi_pie_001_neutral.jpg
│   │   │   └── ...
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── angry/
│   │   ├── surprise/
│   │   ├── fear/
│   │   └── disgust/
│   ├── 832x1248/
│   │   ├── neutral/
│   │   └── ...
│   └── ...  (64 bucket directories)
│
├── final_data_raf_bucket/          # RAF synthesized pairs
│   ├── 256x256/
│   │   ├── neutral/               # Real neutral images (super-resolved)
│   │   │   ├── raf_test_0001.jpg
│   │   │   └── ...
│   │   ├── happy/                  # API-generated expressions
│   │   ├── sad/
│   │   ├── angry/
│   │   ├── surprise/
│   │   ├── fear/
│   │   └── disgust/
│   └── ...  (64 bucket directories)
│
├── scripts/
│   ├── stat_resolution.py          # Resolution statistics
│   ├── allocate_buckets.py         # Bucket allocation
│   ├── seedvr2_upscale_buckets.py  # SeedVR2 super-resolution
│   ├── resize_to_buckets.py        # Final resize
│   ├── generate_raf_expressions.py # API expression generation
│   └── verify_buckets.py           # Verification
│
└── README.md
```

---

## Usage

### Loading the Dataset

```python
from pathlib import Path
from PIL import Image

dataset_root = Path("final_data_v1_bucket")

# Iterate over all buckets
for bucket_dir in sorted(dataset_root.iterdir()):
    if not bucket_dir.is_dir():
        continue
    w, h = map(int, bucket_dir.name.split('x'))
    
    # Load paired expressions for each person
    neutral_dir = bucket_dir / "neutral"
    for img_path in sorted(neutral_dir.glob("*.jpg")):
        neutral = Image.open(img_path)
        
        # Load corresponding expressions
        for expr in ["happy", "sad", "angry", "surprise", "fear", "disgust"]:
            expr_path = bucket_dir / expr / img_path.name
            if expr_path.exists():
                expr_img = Image.open(expr_path)
```

### Aspect Ratio Bucketing for Training

For Flux fine-tuning, use aspect ratio bucketing to batch images of the same resolution:

```python
from torch.utils.data import DataLoader, Sampler

class BucketBatchSampler(Sampler):
    """Groups same-bucket images into batches."""
    def __init__(self, bucket_indices, batch_size):
        self.buckets = bucket_indices  # {bucket_name: [indices]}
        self.batch_size = batch_size
    
    def __iter__(self):
        for bucket, indices in self.buckets.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
```

---

## Technical Details

### Flux Architecture Constraints

The Flux DiT architecture imposes specific requirements on input dimensions:

```
Input Image (H × W)
    │
    ▼
VAE Encoder (8× downsample)
    │  → Latent: (H/8 × W/8 × C)
    ▼
Patchify (2×2 patches)
    │  → Tokens: (H/16 × W/16) tokens
    ▼
Flatten + RoPE Position Encoding
    │  → Sequence length L = (H × W) / 256
    ▼
DiT Transformer Blocks
    │
    ▼
Unpatchify + VAE Decoder → Output Image (H × W)
```

**Constraint derivation**:
- VAE: requires H, W divisible by 8
- Patchify: requires latent H/8, W/8 divisible by 2 → H, W divisible by 16
- Sequence packing: requires (H/8)/2 and (W/8)/2 to be integers → same as above

### Super-Resolution Model

**SeedVR2-7B** (Wang et al., ICLR 2026) was chosen for the following reasons:

1. **One-step inference**: Unlike multi-step diffusion SR models, SeedVR2 produces results in a single forward pass, enabling efficient batch processing of ~4,752 images
2. **Arbitrary resolution**: No fixed input/output resolution constraints, unlike GFPGAN (fixed 512×512 face prior) or Real-ESRGAN (fixed 4× scale)
3. **Identity preservation**: Adversarial post-training against real data ensures generated textures are realistic and identity-consistent
4. **7B parameter capacity**: Large model capacity captures fine-grained facial features critical for expression recognition

### API Expression Generation

The doubao-seedream API performs instruction-based image editing, preserving identity while modifying expressions. Key advantages over alternative approaches:

- **vs. GAN-based expression transfer** (e.g., StarGAN): Higher visual quality, better identity preservation
- **vs. 3DMM-based rendering**: More photorealistic, handles diverse face shapes naturally
- **vs. diffusion-based img2img**: Instruction-following enables precise expression control

---

## Processing Scripts

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| `stat_resolution.py` | Analyze resolution distribution | `--data_dir`, `--name` |
| `allocate_buckets.py` | Assign people to buckets | `--v1_dir`, `--raf_dir`, `--v1_out`, `--raf_out` |
| `seedvr2_upscale_buckets.py` | Batch super-resolution | `--bucket_dir`, `--seedvr_dir`, `--num_gpus` |
| `resize_to_buckets.py` | Final pixel-exact resize | `--bucket_dir`, `--backend` |
| `generate_raf_expressions.py` | Synthesize RAF expressions | `--input_dir`, `--output_dir`, `--prompt_dir` |
| `verify_buckets.py` | Validate allocation correctness | `--v1_bucket`, `--raf_bucket` |

---

## License

- **Multi-PIE**: Subject to CMU Multi-PIE license
- **KDEF**: Subject to KDEF license (research use only)
- **Oulu-CASIA**: Subject to University of Oulu license
- **RAF-DB**: Subject to RAF-DB license (research use only)
- **Processing scripts**: MIT License

Please ensure compliance with individual dataset licenses before use.

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{multi_res_face_expression_2026,
    title={Multi-Resolution Facial Expression Paired Dataset for Flux Fine-Tuning},
    year={2026},
    note={Constructed from Multi-PIE, KDEF, Oulu-CASIA, and RAF-DB with SeedVR2 super-resolution}
}
```

### Acknowledgments

- **SeedVR2**: Wang et al., "SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training", ICLR 2026
- **Flux**: Black Forest Labs, "FLUX.1 and FLUX.2" family of diffusion models
- **Multi-PIE**: Gross et al., "Multi-PIE", Image and Vision Computing, 2010
- **KDEF**: Lundqvist et al., "The Karolinska Directed Emotional Faces", 1998
- **Oulu-CASIA**: Zhao et al., "Facial Expression Recognition from Near-Infrared Videos", 2011
- **RAF-DB**: Li et al., "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild", CVPR 2017
