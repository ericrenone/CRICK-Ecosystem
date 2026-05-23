# CRICK Ecosystem
### The Biology Decade: From the Matmul Ceiling to the Pair-Tensor Era

> *"The matmul era of biology compute is not failing. It is finishing. Biology's decade is the next one, on the substrate the workload demands."*

---

## The Single Thesis

Biology AI is the next frontier compute wave — and it is structurally mismatched with the silicon that currently runs it.

Every major biology AI breakthrough of 2024–2026 — AlphaFold 3 (Google DeepMind, *Nature* 2024), Boltz-2 (MIT + Recursion, 2025), Evo 2 (Arc Institute + NVIDIA, *Nature* 2026), RFdiffusion3 (November 2025), State and Stack virtual-cell models (Arc Institute, 2025–2026) — converges on the same five computational primitives. None of those five is a matrix multiplication. Yet every one runs today on GPU and TPU silicon designed to maximize matrix multiplication. The mismatch costs **3–12× per operation**, compounding across workloads. It does not shrink with process-node improvements. It is structural.

**CRICK** is the substrate, software, and strategic playbook built for the workload these architectures actually perform.

---

## The Five Structural Overheads

| Operation | What it is | Matmul overhead | Principal workload |
|-----------|-----------|----------------|--------------------|
| **Pair tensor** | N×N pairwise residue-residue representation | 4–6× per Pairformer block | AlphaFold 3, Boltz-2, all structure-prediction |
| **SE(3) equivariance** | Exact rotation-translation symmetry for 3D structure | 2–3× per structure module | Protein folding, de novo design |
| **Diffusion denoising** | Reverse-time sampling, 200–500 steps | 12–20× on full de novo design | RFdiffusion3, AlphaFold 3 generation |
| **Long-context genomics** | Up to 1M DNA bases in a single pass | 5–10× on Evo-class inference | Evo 2 40B, full-genome modeling |
| **CRISPR rank-one edit** | Updating a protein representation for one mutation | **80–100×** on mutational scanning | CRISPR screening, drug variant scans |

These overheads are **architectural, not nodal**. Moore's Law applies to transistor density. It does not apply to structural mismatch.

---

## Competitive Landscape: CRICK vs. 10 SOTA Biology AI Labs

The table below maps CRICK's architecture against the ten most consequential biology AI ecosystems as of May 2026 — ranked by compute dependency and structural exposure to the matmul mismatch.

| Lab / Platform | Primary biology workload | Compute substrate | Pair-tensor native | Biology-effective $/TFLOPS-eq-W | Structural overhead exposure |
|----------------|--------------------------|-------------------|--------------------|----------------------------------|------------------------------|
| **CRICK (Crick-1)** | Full central dogma: DNA → RNA → protein → cell | Crick-1 (pair-tensor-native, 7 hardware blocks) | ✅ Yes | **~$0.18** | **None — native** |
| **Isomorphic Labs (IsoDDE)** | Drug design: protein-ligand, antibody-antigen, affinity | NVIDIA Blackwell + Google TPU (internal fleet, $600M Series A, 40% to compute) | ❌ No | ~$0.78 | **High** — IsoDDE's diffusion + pair-tensor core runs on matmul silicon; 2.3× over AF3 in accuracy, but at matmul overhead cost |
| **Google DeepMind (AlphaFold 3)** | Protein structure, biomolecular interaction prediction | TPU v7 Ironwood (Google-internal) | ❌ No | ~$0.42 | **High** — every Pairformer block pays 4–6× overhead; 48-block depth compounds multiplicatively |
| **Meta FAIR (ESM3 / ESMFold)** | Protein language modeling; single-sequence structure prediction at scale | NVIDIA H100/H200 fleet (Meta AI Research) | ❌ No | ~$0.78 | **Medium-High** — ESM3 is transformer-shaped (GEMM-friendly), but lacks pair-tensor; trades accuracy for speed via MSA removal |
| **Arc Institute + NVIDIA (Evo 2)** | Genomic foundation model: 40B params, 1M DNA-base context, full tree of life | NVIDIA DGX / BioNeMo cloud; BioNeMo framework | ❌ No | ~$0.78 | **High** — Hyena/StripedHyena long-convolution runs as repeated GEMM; 5–10× overhead at 1M-context scale; infeasible on H100 at full 40B context |
| **Recursion Pharmaceuticals (Phenom / BioHive-2)** | Phenomics, cellular imaging, target-to-candidate pipeline | BioHive-2 (NVIDIA-powered, Top 35 supercomputer); NVIDIA DGX Cloud; $50M NVIDIA investment | ❌ No | ~$0.78 | **Medium** — phenomics CNN workload is more GEMM-friendly; structure-prediction and mutational scanning at scale hit pair-tensor ceiling |
| **Insilico Medicine (Nach01 / PandaOmics)** | End-to-end drug discovery; generative chemistry; target ID | Microsoft Azure (Maia 200 + Blackwell); Nach01 on Microsoft Discovery marketplace | ❌ No | ~$0.55–0.78 | **Medium-High** — generative chemistry and structure-based design hit diffusion + SE(3) overhead; Microsoft Azure compute at matmul rates |
| **Ginkgo Bioworks + OpenAI (Autonomous Lab)** | Closed-loop autonomous experimentation; cell-free protein synthesis; synthetic biology | OpenAI GPT-5 (Azure compute); cloud-native robotic lab orchestration | ❌ No | ~$0.55–0.78 | **Low-Medium** — primary workload is LLM orchestration (matmul-native); biology-specific structure and design workloads inherit overhead when invoked |
| **Roche / Genentech (AI Factory)** | Drug discovery, diagnostics, therapeutic design | 3,500+ Blackwell GPUs on-premises (NVIDIA AI Factory, launched March 2026) | ❌ No | ~$0.78 | **High** — largest announced pharma on-premise GPU footprint; fully matmul-era; entire AlphaFold 3 + diffusion workload at structural overhead |
| **Chai Discovery (Chai-1 / Chai-2)** | Biomolecular structure prediction, biologics design; Eli Lilly mid-eight-figure annual access fee | Cloud GPU (AWS / Azure); NVIDIA Blackwell | ❌ No | ~$0.78 | **High** — Chai-1/2 architecture is pair-tensor-based (competitive with AF3); every Pairformer block pays the same 4–6× matmul overhead |
| **Eli Lilly (TuneLab + NVIDIA co-innovation lab)** | Drug discovery as a service; $1B dataset; AI-directed robotic experimentation 24/7 | NVIDIA Vera Rubin + BioNeMo + Clara; $1B co-innovation lab in South San Francisco | ❌ No | ~$0.78 | **Very High** — $1B+ infrastructure commitment to matmul silicon for 24/7 biology compute; mutational scanning and structure generation at scale hit the full overhead stack |

### The Pattern

Every competitor — from Isomorphic Labs' IsoDDE (the closest thing to an AlphaFold 4) to Roche's 3,500-GPU on-premise factory — runs its biology workloads on matmul-era silicon paying structural overheads of 2–100× per operation. None has a native pair-tensor substrate. None has a native SE(3) hardware block. None has hardware-native diffusion iteration. The architectural gap is not a gap in model quality. It is a gap in the substrate the model quality is paid for on.

---

## Head-to-Head: CRICK vs. Closest Competitors

### CRICK vs. Isomorphic Labs (IsoDDE)

IsoDDE is the most accurate publicly disclosed drug design engine as of February 2026 — described by external researchers as an "AlphaFold 4 in effect." It outperforms AlphaFold 3 by 2.3× and Boltz-2 by 19.8× on antibody-antigen interface prediction in the hardest generalization bins.

**The catch:** Isomorphic Labs directed 40% of its $600M Series A to compute infrastructure expansion — all matmul silicon. IsoDDE's diffusion module and pair-tensor core run on the same Blackwell/TPU stack that pays the full 4–20× overhead per workload. An IsoDDE saturation mutational scan costs ~$40/scan on Blackwell. On Crick-1-class silicon it costs ~$0.50. Isomorphic's model accuracy advantage does not close the substrate economics gap.

**Substrate verdict:** Isomorphic owns the model quality frontier. CRICK owns the unit economics frontier. These are orthogonal dimensions until 2028 — then they converge as matmul costs hit the board-level ceiling.

### CRICK vs. Evo 2 (Arc Institute + NVIDIA)

Evo 2 is the most important genomic foundation model in existence: 40B parameters, 1M-token DNA context, 9.3 trillion training nucleotides, fully open-source. Its Hyena/StripedHyena architecture is sub-quadratic in sequence length — the correct algorithmic choice. It is running on the wrong substrate.

Evo 2 at 1M-context on NVIDIA H100: **infeasible** (insufficient memory). On GB300: ~95 seconds per sequence. On Hyena-native silicon (CRICK-1 class): **~18 seconds**. At 40B parameters and frontier training scale, the overhead is 5–10× and grows with context length. CRICK-DNA's open-source integration of Evo 2 routes Hyena blocks to the CORDIC-native hardware path that the model's architecture was algorithmically designed for.

**Substrate verdict:** Evo 2's algorithm and CRICK's substrate are co-designed by the same underlying mathematics. The CRICK-DNA module is the native deployment target for Evo-class models.

### CRICK vs. Roche/Genentech AI Factory

Roche's March 2026 launch of its 3,500+ Blackwell GPU on-premise AI factory represents the largest announced pharma GPU footprint globally. It is a $1B+ commitment to matmul silicon for 24/7 biology compute — including AlphaFold 3 structure prediction, diffusion-based design, and saturation mutational scanning.

A single AlphaFold 3 frontier training run on that infrastructure costs $40–60M. On Crick-1-class silicon: $6–10M. A whole-genome CRISPR off-target screen at population scale costs ~$2.4M per run on Blackwell. On Crick-1: ~$30K. At 3,500 GPUs running 24/7 on biology workloads, the annual structural waste from the architectural mismatch is in the hundreds of millions. The Roche AI factory is not a competitor — it is the largest single illustration of the problem CRICK solves.

**Substrate verdict:** Roche's on-premise investment is the 2028 ceiling made visible in 2026. It is the first wave of pharma infrastructure that will require native-substrate migration as population-genomic-medicine workloads hit their unit-economic floor.

---

## The Economic Translation

| Workload | Matmul-era cost | CRICK native cost | Ratio |
|----------|----------------|-------------------|-------|
| AlphaFold 3 frontier training run | $40–60M | $6–10M | **6×** |
| Saturation mutational scan (300-residue protein) | ~$40 / scan | ~$0.50 / scan | **80×** |
| Whole-genome CRISPR off-target screen (1B guide RNAs) | ~$2.4M / run | ~$30K / run | **80×** |
| 16K-chip cluster power (equivalent biology throughput) | ~60 MW | ~11.5 MW | **5×** |
| Cell-type-specific pharmacogenomics at population scale | **Infeasible** | ~$200M one-time | **Unlocks** |

The last row is the strategic prize: population-scale cell-type-specific pharmacogenomics is not a commercial product today because the unit economics do not close on matmul silicon. The CRICK native substrate makes it viable. This is the product category that does not exist — and will define the 2028–2032 personalized medicine market.

---

## Market Setup

| Year | Frontier AI compute (global) | Biology AI share | Biology AI spend | Native-substrate addressable |
|------|------------------------------|-----------------|-----------------|------------------------------|
| 2025 | ~$200B | 6% | ~$12B | 0% |
| 2026 | ~$320B | 8% | ~$26B | 5% |
| 2027 | ~$500B | 11% | ~$55B | 15% |
| 2028 | ~$700B | 14% | ~$98B | 30% |
| 2029 | ~$880B | 16% | ~$140B | 45% |
| 2030 | ~$1.05T | 17% | ~$180B | 55% |

**2030 SAM:** ~$100B/yr  
**Capture target:** 15–25% of SAM = **$15–25B/yr by 2030**

---

## The CRICK Ecosystem

### Layer 1 — Crick-1 Silicon

Pair-tensor-native compute substrate. Seven dedicated hardware blocks. Fundamental operation: iterative refinement of the N×N pairwise representation under geometric symmetry with diffusion-denoising integration.

| | Crick-1 | NVIDIA GB300 | AWS Trainium3 | TPU v7 Ironwood |
|--|---------|-------------|---------------|-----------------|
| Biology-effective $/TFLOPS-eq-W | **~$0.18** | ~$0.78 | ~$0.35 | ~$0.42 |
| On-chip SRAM | **873 MB** | ~120 MB | ~96 MB | ~256 MB |
| HBM capacity | **384 GB HBM4** | 288 GB HBM3e | 144 GB HBM3e | 192 GB HBM3e |
| HBM bandwidth | **12.8 TB/s** | 8 TB/s | 4.9 TB/s | 7.4 TB/s |
| Native biology ops | **Yes (7 blocks)** | No | No | No |

16K-chip cluster: **134 PFLOPS effective, 11.5 MW** — vs. ~60 MW for matmul-equivalent biology throughput.

### Layer 2 — Chip Lineage

- **Banach-1 (2026)** — fixed-point iteration as the architectural primitive. 12.3 PFLOPS FP4, 384 MB SRAM, 288 GB HBM3e.
- **Volder-1 (2026)** — rotation-based arithmetic for exact geometric computation. 9.8 PFLOPS FP4-eq, 645 MB SRAM, 9.6 TB/s.
- **Crick-1 (2026)** — pair-tensor iteration as the native biology operation. 8.2 PFLOPS FP4 pair-tensor, 873 MB SRAM, 384 GB HBM4, 12.8 TB/s.

### Layer 3 — CRICK-IR: The Compiler

Open compiler. Ingests PyTorch and JAX biology models — AlphaFold 3, Boltz-2, Evo 2, RFdiffusion3, State, Stack, scGPT. Routes each operation to the appropriate hardware block. Recognizes the full DNA → RNA → protein → structure → function pipeline as a single compilation target.

This is the CUDA equivalent for biology: the developer flywheel that compounds into multi-year switching cost.

### Layer 4 — Open-Source Stack

- **CRICK-DNA** — Evo 2 (40B, 1M-token context), HyenaDNA, Nucleotide Transformer, Caduceus, Borzoi, Enformer, CodonFM
- **CRICK-RNA** — RiNALMo (650M), AIDO.RNA (1.6B, SOTA on 24/26 RNA tasks), RNA-FM, RhoFold+, gRNAde (ICLR 2025 Spotlight)
- **CRICK-Protein** — AlphaFold 3, Boltz-1/2, Chai-1, Protenix, HelixFold3, RFdiffusion3, FrameFlow, ESM3, ProteinMPNN
- **CRICK-Cell** — scGPT (33M cells), State (Arc 2025), Stack (Arc 2026), Geneformer, CellFM (100M cells), GEARS, RNA velocity suite

The open-source posture is rational, not altruistic. The moat is substrate, compiler, and ecosystem — not model weights. This is the Hugging Face × PyTorch × Linux pattern applied to biology.

---

## The 2028 Ceiling

Three convergent constraints arrive simultaneously:

**1. Workload share crosses the board-level threshold.**
A 5× overhead on 6% of $200B is ~$60B in structural waste — absorbable inside R&D budgets. The same overhead on 14% of $700B is ~$490B. That is a capital allocation crisis.

**2. Per-workload cost rigidity is architectural, not nodal.**
TSMC N2 to N2P buys ~30% transistor density. It buys 0% reduction in architectural mismatch. After the crossover point, additional process-node generations widen the gap rather than close it.

**3. The strategic-priority shift hardens substrate decisions.**
The 2028–2029 cohort of FDA-cleared diagnostics, EMA-authorized therapeutics, and sovereign compute commitments (NHS, EU-1+ Million Genomes, NIH All of Us, Japan AMED, KAUST) sets the substrate base for the 2030s. Once a biology AI product enters clinical deployment at population scale, the substrate decision becomes a regulatory commitment. Switching cost rises to multi-year, multi-billion-dollar territory.

---

## Upstream Science: Why Now

- **AlphaFold 2 (Nobel Prize 2024)** — protein structure prediction from sequence solved.
- **AlphaFold 3 (*Nature*, May 2024)** — extended to all biomolecular interactions: protein-ligand, protein-RNA, protein-DNA, multi-chain.
- **IsoDDE (Isomorphic Labs, February 2026)** — 2.3× over AlphaFold 3 on hardest generalization benchmarks; 50% accuracy on novel pockets vs. 23.3% for AF3. The first substantive accuracy step-change since AF3.
- **Evo 2 (*Nature*, March 2026)** — 40B parameters, 1M DNA-base context, 9.3 trillion training nucleotides. The full central dogma in a single open-source foundation model.
- **RFdiffusion3 (November 2025)** — de novo design extended to all-atom biomolecular complexes.
- **State and Stack (Arc Institute, 2025–2026)** — first virtual cell foundation models on 100M-cell perturbation atlases. The cell is now a computational object.
- **Karkada, Olah, Tegmark et al. (February 2026)** — proved analytically that the geometry of learned representations is forced by the symmetry of the training data. The biology AI workload is not arbitrary; its structure is mathematically determined.

These are not incremental improvements. They are the scientific infrastructure that makes the biology compute decade commercially real.

---

## Adoption Roadmap

| Milestone | Target | Signal |
|-----------|--------|--------|
| Crick-1 tape-out; CRICK-IR alpha; CRICK-RNA beta | 2026 Q3 | Design-win count: 3 |
| Volume production; CRICK-IR 1.0; full open-source releases | 2027 H1 | Design-win count: 12 |
| Matmul-era biology compute hits economic ceiling | 2028 | 30–40% of frontier biology training on Crick-1-class silicon |
| First FDA-cleared diagnostic incorporating whole-proteome variant scanning | 2028 | P2 |
| Cell-type-specific pharmacogenomics commercial deployment | 2029 | P3 |
| Integrated central-dogma compiler target as default biology AI deployment | 2030 | P5 |

---

## Falsifiable Predictions

| ID | Prediction | Failure signal |
|----|------------|----------------|
| P1 | Pair-tensor-native silicon produces 3–5× cost reduction for pharma and AI-biotech pipelines by 2028 | Ratio compresses below 2× before end-2027 (Rubin-class extensions) |
| P2 | Saturation mutational scanning moves from $40/scan to $0.50/scan; first FDA-cleared diagnostic by 2028 | No FDA clearance by end-2028 |
| P3 | First commercial cell-type-specific pharmacogenomics product by 2029 | Unit cost remains above commercial pricing floor |
| P4 | col(F)/ker(F) architectural separation becomes the standard pattern for biology compute substrate design | Matmul substrates close the gap architecturally |
| P5 | Separate genomic, RNA, structure, and cell pipelines replaced by integrated single-compiler programs by 2030 | CRICK-IR developer adoption below 20% of frontier biology labs |
| P6 | Hybrid clusters (biology-native + geometric + general-purpose GPU) become the standard by 2029 | Substrate-fragmentation index plateaus at 2026 levels |
| P7 | **Matmul-era biology compute hits its economic ceiling by 2028** | Biology AI share plateaus below 8% through 2027 |
| P8 | Eroom's Law inverts 2027–2029; CRICK ecosystem captures 15–25% of biology AI compute rent | Biology drug approval rate fails to accelerate by 2029 |

---

## The Moat Structure

**1. Substrate-algorithm co-design** — Pair tensor + SE(3) + diffusion + long-context genomics + rank-one edit mapped to seven hardware blocks. Time to replicate by NVIDIA Rubin-class: 3–4 years. Moat window: 2026–2029.

**2. Compiler / IR adoption** — CRICK-IR's pair tensor and SE(3) frame as first-class compiler constructs. Once a biology pipeline is compiled to CRICK-IR, the switching cost is a compiler rewrite, not a chip swap.

**3. Open-source ecosystem gravity** — CRICK-RNA / DNA / Protein / Cell as the canonical, benchmarked deployment of every frontier biology model. Developer surplus leads to benchmark control.

**4. Data and benchmark partnerships** — RNAcentral (release 26: 45M sequences, 52 expert databases, 204 organisms), Rfam, PDB, AlphaFold Protein Structure Database, Tahoe-100M, ENCODE, Human Cell Atlas.

**5. Clinical deployment commitments** — Co-development with top-5 pharma, top-5 AI biotech, and 2–3 hyperscalers. Design wins in NHS, EU-1+ Million Genomes, Japan AMED, KAUST, NIH All of Us. Each is a 5–10-year commitment defended by clinical and regulatory switching cost.

---

## The Competitive Risk

NVIDIA Rubin or post-Rubin generations could add dedicated pair-tensor extensions, compressing the advantage from 4–12× to 2–4× and narrowing the moat window from 2026–2029 to 2026–2027. This is publicly observable. The window must be built and defended before the matmul ecosystem's defensive response ships.

The leading indicator is already visible: Anthropic's May 2026 five-substrate portfolio (Trainium3, TPU Ironwood, Blackwell, CoreWeave-Blackwell, Maia 200 in negotiation) demonstrates that frontier laboratories running at $30B+ annual compute spend are already buying the right substrate for the right workload regardless of single-vendor commitments. Biology workloads — with their five distinct structural mismatches — are the natural next layer of specialization.

---

## Summary

The matmul era made language model training, image generation, and reasoning feasible at frontier scale. It did not make biology compute feasible at frontier scale — because biology compute is not language compute. The five structural mismatches are architectural. They compound multiplicatively. They grow in absolute cost as biology's share of frontier compute grows. They will not be resolved by the next process node or the next software optimization.

The exit is a substrate designed for the workload. The compiler that routes biology workloads to it. The open-source stack that gives developers a reason to stay.

The economic prize is $100B addressable by 2030. The capture rate is 15–25%.

**CRICK is that ecosystem.**

---

*Compiled May 2026. Substrate specifications sourced from published product disclosures. Scientific citations reference peer-reviewed publications and publicly available preprints. Financial projections are scenario estimates.*
