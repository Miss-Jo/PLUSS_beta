# PLUSS_β Architecture Design Document

## Core Design Principle

**CRITICAL ARCHITECTURAL REQUIREMENT**: The Semantic Tuner and Box Tuner must maintain **COMPLETELY SEPARATE** computational graphs and gradient flows.

```
┌─────────────────────────────────────────────────────┐
│                   PLUSS_β Framework                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │ Semantic Tuner   │      │   Box Tuner      │    │
│  │                  │      │                  │    │
│  │ Gradient Flow ━┓ │      │ Gradient Flow ━┓ │    │
│  │               ▼ │      │               ▼ │    │
│  │  L_align        │      │  L_box          │    │
│  │  (Eq. 10)       │      │  (Eq. 16)       │    │
│  └──────────────────┘      └──────────────────┘    │
│         │                          │                │
│         │  NO GRADIENT FLOW        │                │
│         │  ════════════════        │                │
│         ▼                          ▼                │
│  Semantic Optimizer        Box Optimizer            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Gradient Isolation Enforcement

### 1. Semantic Tuner Training Path

```python
# ONLY these parameters receive gradients from L_align
semantic_tuner.parameters()
  ├── prompts[0..11]  # Learnable prompts for each layer
  └── (temperature)   # Learnable temperature in loss

# Computational graph:
f_I (detached) → SemanticTuner → f_ST → L_align → ∇ → semantic_optimizer

# What's frozen:
- CLIP image encoder parameters ✓
- CLIP text encoder parameters ✓
- SAM parameters ✓
- Box Tuner parameters ✓✓✓  (CRITICAL)
```

### 2. Box Tuner Training Path

```python
# ONLY these parameters receive gradients from L_box
box_tuner.parameters()
  ├── sam_projection
  ├── cross_attention
  ├── mlp
  └── alpha_{x,y,w,h}  # Refinement scaling factors

# Computational graph:
SAM_tokens (detached) + CLIP_features (detached) 
  → BoxTuner → B_refined → L_box → ∇ → box_optimizer

# What's frozen:
- CLIP parameters ✓
- SAM parameters ✓
- Grounding DINO parameters ✓
- Semantic Tuner parameters ✓✓✓  (CRITICAL)
```

## Training Flow per Iteration

```
Input: Image I, Text T

┌─────────────────────────────────────────────┐
│ Step 1: Extract Features (Frozen CLIP)      │
│ f_I = CLIP.encode_image(I)                  │
│ f_T = CLIP.encode_text(T)                   │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ Step 2: Two-Branch Forward                  │
│                                              │
│ Branch 1 (Point):  CLIP → Points → SAM → z  │
│ Branch 2 (Box):    DINO → Boxes → SAM → ẑ   │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ Step 3: Hard Example Mining                 │
│ L_mask = α·L_IoU + β·L_Dice                 │
│ if L_mask ≥ σ: add to MemoryBank            │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ Step 4: Point-2-Box Conversion              │
│ B_pseudo = DBSCAN_cluster(points)           │
└─────────────────────────────────────────────┘
              ↓
        ╔═══════════╗
        ║  FORK     ║
        ╚═══════════╝
         ↙         ↘
┌──────────────┐  ┌──────────────┐
│ Path A:      │  │ Path B:      │
│ Train Box    │  │ Train Sem.   │
│ Tuner        │  │ Tuner        │
│              │  │ (every 100   │
│ EVERY iter   │  │  epochs)     │
└──────────────┘  └──────────────┘

Path A (Box Tuner) - EVERY iteration:
────────────────────────────────────
1. Extract SAM tokens (detached)
2. Extract CLIP region features (detached from f_I)
3. Fuse through BoxTuner → F_fused
4. Predict delta_B
5. Refine boxes: B_ref = refine(B_init, delta_B)
6. Compute L_box = λ₁·L_L1 + λ₂·L_GIoU
7. box_optimizer.zero_grad()
8. L_box.backward()  # Gradients ONLY to BoxTuner
9. box_optimizer.step()

Path B (Semantic Tuner) - Every 100 epochs:
──────────────────────────────────────────
1. Sample hard_batch from MemoryBank
2. f_ST = SemanticTuner(f_I)  # Apply prompts
3. Compute L_align = -log(exp(cos(f_ST, f_T)/τ) / Σ)
4. semantic_optimizer.zero_grad()
5. L_align.backward()  # Gradients ONLY to SemanticTuner
6. semantic_optimizer.step()
```

## Implementation Details

### Preventing Gradient Leakage

```python
# trainer.py - Box Tuner Training

def train_box_tuner_step(self, ...):
    # CRITICAL: Detach all inputs from semantic path
    clip_feature_map = self.get_clip_feature_map(
        images, 
        f_I.detach()  # ← Detach here!
    )
    
    # Forward through box tuner
    fused_features = self.box_tuner(
        sam_tokens.detach(),  # ← Detach SAM tokens
        clip_region_features   # Already detached via f_I
    )
    
    # Compute loss - gradient graph starts here
    loss, metrics = self.box_loss_fn(B_ref, B_pseudo)
    
    # Optimize ONLY box tuner
    self.box_optimizer.zero_grad()
    loss.backward()  # Gradients flow only to box_tuner.parameters()
    self.box_optimizer.step()
```

```python
# trainer.py - Semantic Tuner Training

def train_semantic_tuner_step(self, hard_batch):
    # Get features from memory bank (already detached)
    f_I = hard_batch['f_I'].to(self.device)
    f_T = hard_batch['f_T'].to(self.device)
    
    # Apply semantic tuner
    f_ST = self.semantic_tuner.get_adapted_features(
        self.clip_model,  # Frozen
        f_I  # Detached
    )
    
    # Compute alignment loss
    loss = self.semantic_loss_fn(f_ST, f_T, ...)
    
    # Optimize ONLY semantic tuner
    self.semantic_optimizer.zero_grad()
    loss.backward()  # Gradients flow only to semantic_tuner.parameters()
    self.semantic_optimizer.step()
```

### Why This Separation?

1. **Semantic Tuner Focus**: Learn to distinguish between visually similar but semantically different objects
   - Examples: Different species of fish, various dog breeds
   - Optimizes for: Better alignment with text descriptions
   
2. **Box Tuner Focus**: Learn to accurately localize objects spatially
   - Examples: Refining tight bounding boxes, handling occlusion
   - Optimizes for: Better spatial overlap with ground truth

3. **Prevents Interference**: 
   - Semantic gradients might harm spatial precision
   - Spatial gradients might harm semantic discrimination
   - Each module can specialize without compromise

## Memory Bank Dynamics

```
Memory Bank Flow:
─────────────────

New sample arrives
      ↓
Compute L_mask between z and ẑ
      ↓
  L_mask ≥ σ?
   ↙     ↘
  No      Yes
  ↓        ↓
Skip    Create entry: e = (f_I, f_T, z, ẑ, L_mask, M_pseudo)
         ↓
    Add to bank
         ↓
    Size > C?
      ↙  ↘
     No   Yes
     ↓     ↓
   Keep  Remove oldest (FIFO)
```

## Inference Pipeline (Algorithm 3)

```
Input: Image I, Text T

┌────────────────────────────────────┐
│ Load trained: SemanticTuner,      │
│               BoxTuner             │
└────────────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ f_I = CLIP.encode_image(I)        │
│ f_ST = SemanticTuner(f_I)         │
│ f_T = CLIP.encode_text(T)         │
└────────────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ B_init = GroundingDINO(I, T)      │
└────────────────────────────────────┘
              ↓
    ┌─────────────────┐
    │ For each box b: │
    └─────────────────┘
         ↓
    ┌──────────────────────────────┐
    │ S_b = SAM.tokens(b)          │
    │ F_b = CLIP.RoIAlign(b)       │
    │ F_fused = BoxTuner(S_b, F_b) │
    │ Δb = predict(F_fused)        │
    │ b_refined = refine(b, Δb)    │
    └──────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ M_final = SAM(I, {b_refined}, T)  │
└────────────────────────────────────┘

Note: Uses ONLY box-prompt branch
      Point-prompt branch not used in inference
```

## Key Equations

### Mask Loss (Hard Example Mining)

```
L_IoU(z, ẑ) = 1 - (Σ z_i·ẑ_i) / (Σ(z_i + ẑ_i - z_i·ẑ_i))     (Eq. 3)

L_Dice(z, ẑ) = 1 - (2·Σ z_i·ẑ_i) / (Σ z_i² + Σ ẑ_i²)          (Eq. 4)

L_mask = α·L_IoU + β·L_Dice     (α=0.7, β=0.3)                 (Eq. 5)
```

### Semantic Tuner Loss

```
L_align = -log(
    exp(cos(f_ST(i), f_T(i))/τ) / 
    Σ_j exp(cos(f_ST(i), f_T(j))/τ)
)                                                               (Eq. 10)
```

### Box Tuner Fusion

```
F_fused = softmax((F_S·K_F^T) / √C) · V_F                      (Eq. 11)
```

### Box Refinement

```
B_ref = (
    x + α_x·Δx,
    y + α_y·Δy,
    w·exp(α_w·Δw),
    h·exp(α_h·Δh)
)                                                               (Eq. 13)
```

### Box Loss

```
L_L1 = ||B_ref - B_pseudo||_1                                  (Eq. 14)

L_GIoU = 1 - GIoU(B_ref, B_pseudo)                            (Eq. 15)

L_box = λ_L1·L_L1 + λ_GIoU·L_GIoU                             (Eq. 16)
```

## Parameter Counts

Typical configuration (ViT-B/16 CLIP):
```
Semantic Tuner:  ~98K parameters
  - 12 layers × 16 prompts × 512 dim = 98,304

Box Tuner:       ~2.1M parameters
  - SAM projection: 256→512
  - Cross-attention: 512×512×8 heads
  - MLP: 512→2048→1024→4
  - Scaling factors: 4
  
Total trainable: ~2.2M parameters
(vs ~150M frozen in CLIP, ~630M in SAM)
```

## Conclusion

The architectural separation between Semantic and Box Tuners is **not optional** - it is a fundamental design principle that ensures each module can specialize effectively without interference. Always verify gradient isolation when modifying the code.
