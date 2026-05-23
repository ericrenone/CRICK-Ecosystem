# CRICK — Ecosystem

*The Central Dogma in Silicon, in Software, in Strategy. A unified stack from the col(F)/ker(F) genetic-boundary mathematics through the pair-tensor-native silicon through the canonical biology AI software to the unit economics of population-scale genomic medicine.*

---

## The One-Page Thesis

Biology is a one-way col(F)/ker(F) information system. DNA → RNA → protein → function is the directed flow; the genome is col(F); the epigenome, the wobble degeneracy of the genetic code, and the conformational microstates of folding are ker(F). The deep-learning architectures that have come to dominate biology — AlphaFold 3, Boltz-2, Chai-1, Protenix, HelixFold3, RFdiffusion3, ESM3, RhoFold+, RNA-FM, RiNALMo, AIDO.RNA, Evo 2, State, Stack, scGPT, Geneformer — are the operational realization of that flow, all converging on the same intermediate data structure: the N×N×D pair tensor refined under SE(3)-equivariant triangle attention with diffusion-denoising integration.

Those workloads are running on substrates designed for language-model matrix multiplication. The pair tensor is packed into 2D matmul tiles; SE(3) rotations are emulated by polynomial Taylor expansions in a separate vector unit; diffusion sampling is repeated full forward passes through HBM; long-context Hyena is approximated as long convolutions on the matmul path; CRISPR rank-one updates are handled as full re-inferences. Each is an emulation. Each pays a 3–12× overhead vs. the natural cost of the underlying operation. The frontier biology training run costs $40–60M in compute; the inference price of a saturation mutational scan is ~$40; cell-type-specific genomic medicine sits below the cost line for population deployment. All three numbers are limits of the substrate, not the science.

**CRICK is the substrate, the software stack, and the strategic playbook for the biology decade.** Crick-1 is the silicon — pair-tensor-native, SE(3)-equivariant, diffusion-iterating, with dedicated hardware for the seven biology-specific primitives the CRICK framework identifies. CRICK-RNA / CRICK-DNA / CRICK-Protein / CRICK-Cell are the canonical open-source software modules covering the full central-dogma pipeline. CRICK-IR is the MLIR-dialect compiler that lowers PyTorch / JAX biology models onto Crick-1. The economic consequence: a 4–12× per-chip advantage on standard biology workloads (AlphaFold 3, Boltz-2, Evo 2, RNA structure), 20–100× on rank-one-update workloads (CRISPR screening, saturation scans, cell-type-specific expression), at ~11.5 MW per 16K-chip cluster vs. ~60 MW for matmul-equivalent throughput. Pharma R&D productivity, currently in the late stages of Eroom’s Law, is the principal beneficiary.

---

## The Stack

The ecosystem is six layers, each with a defined interface to the layer above and below.

### Layer 0 — Mathematical foundations

Seven canonical objects, each over a century or half-century old, each newly load-bearing:

- **Watson–Crick pairing (1953)** — the pair tensor as the native data structure of biology. The double helix is a 2-D pair tensor with a 4/16 sparsity prior; AlphaFold’s Pairformer is its functional generalization.
- **Anfinsen (1972, Nobel)** — sequence determines structure: structure is the col(F) extracted from the sequence; folding is the projection that collapses ~10³⁰⁰ Levinthal microstates to a native ensemble in milliseconds.
- **Crick wobble (1966)** — the 64→20 codon surjection with 44 dimensions of synonymous-codon ker(F): nature’s error-correcting code, with the most degeneracy where mutational pressure is highest.
- **CRISPR-Cas9 (Doudna & Charpentier, 2014, Nobel 2020)** — the programmable Sherman–Morrison rank-one update on the genomic Fisher matrix; one position changed, the rest of the 3.2-billion-base genome intact.
- **Banach (1922)** — the contraction-mapping fixed-point theorem; AlphaFold recycling and reverse-time diffusion SDE integration are its operational realizations.
- **Sherman–Morrison (1950)** — rank-one matrix inverse update; the closed-form linear algebra of a single CRISPR edit on a pair-tensor representation.
- **Volder (1959) + Walther (1971) CORDIC** — shift-and-add iterative arithmetic for rotations, transcendentals, hyperbolic functions, and FFT; SE(3) equivariance and Hyena long-convolution are its operational realizations.

The framework’s identification: the central dogma is the deepest col(F)/ker(F) system in biology; the genome is col(F); the epigenome is heritable ker(F); the chip’s job is to make col(F) extraction operationally feasible at scale.

### Layer 1 — Silicon: the Banach-1 → Volder-1 → Crick-1 lineage

Three chips, three primitive identifications, one substrate philosophy.

- **Banach-1 (2026)** — promotes fixed-point iteration to the architectural primitive on a matmul substrate. The Fixed-Point Compute Unit (FPCU), Anderson Acceleration Block (AAB), and Contraction Monitor (CM) turn AlphaFold recycling, deep equilibrium models, and reverse-time SDE integration into first-class hardware ops. 384 MB SRAM, 288 GB HBM3e, 12.3 PFLOPS FP4.
- **Volder-1 (2026)** — collapses the matmul into the CORDIC iteration. Rotation is the universal computational operation; circular-mode CORDIC handles SE(3) rotations and attention; hyperbolic-mode handles Lorentz, Möbius, and Poincaré-ball geometries. 645 MB SRAM, 9.6 TB/s HBM3e, 9.8 PFLOPS FP4-equivalent on the CORDIC datapath. FFT is a CORDIC operation by construction.
- **Crick-1 (2026)** — adds the **Pair-Tensor Iteration Unit (PTIU)** as the native operation for biology. 128 PTIUs on a TSMC N2P (2nm) die, ~220 B transistors, ~840 mm², CoWoS-L with HBM4. Seven biology-specific hardware blocks: **TAE** (Triangle Attention Engine), **SFB** (SE(3) Frame Update Block), **DDU** (Diffusion Denoiser Unit), **HFE** (Hyena Filter Engine), **SMEU** (Sherman-Morrison Edit Unit), **WCC** (Wobble Code Cache, 4 GB on-chip), **MMC** (Methylation Marker Cache, 3 GB on-chip). 873 MB on-chip SRAM (the largest of any non-wafer-scale accelerator), 384 GB HBM4 at 12.8 TB/s, 4.0 TB/s Ethernet-native scale-up.

Per 16K-chip cluster: 134 PFLOPS-eq pair-tensor FP4, 6.29 PB HBM4 aggregate, 14.3 PB on-chip SRAM, 11.5 MW cluster power. Hybrid deployment is the operating model — Crick-1 for biology, Volder-1 for geometric / hyperbolic workloads, Banach-1 for the Euclidean transformer bulk, all sharing the Ethernet scale-up fabric.

### Layer 2 — CRICK-IR: the compiler

MLIR dialect extending the Volder Intermediate Representation (VIR) with three first-class IR constructs: the **pair tensor**, the **SE(3) frame**, and the **col(F)/ker(F) tensor attribute**. The compiler ingests standard PyTorch / JAX biology models (AlphaFold 3, Boltz-2, RFdiffusion3, Evo 2, RiNALMo, State, Stack, scGPT) and lowers them to PTIU invocations with explicit equivariance, iteration depth, and convergence parameters. The central biology-specific optimization is **central-dogma pipeline lowering** — recognizing the DNA → RNA → protein → structure → function multi-stage workflow at the IR level and scheduling inter-stage data movement to minimize HBM traffic.

Three new operating modes beyond the seven inherited from Banach-1 / Volder-1:
- **Mode H — Pair-Tensor Native** (AlphaFold 3 / Boltz-2 / Chai-1 / Protenix / HelixFold3 class).
- **Mode I — SE(3) Equivariant** (RFdiffusion, RFdiffusion3, FrameFlow, FoldFlow).
- **Mode J — Long-Context Genomic** (Evo 2 7B / 20B / 40B at up to 1 M-token context per chip).

### Layer 3 — Foundation models (canonical, open or open-weights-tracked)

The CRICK ecosystem exposes the full state-of-the-art across the four canonical biology modalities.

- **DNA / Genomic**: Evo 2 (40 B, 1 M-context, *Nature* 2026), Evo, HyenaDNA, Nucleotide Transformer, Caduceus, Borzoi, Enformer, CodonFM (Arc + NVIDIA, 2026).
- **RNA**: RNA-FM, RhoFold+, NuFold, DRfold/DRfold2, trRosettaRNA, RiNALMo (650 M), AIDO.RNA (1.6 B, SOTA on 24/26 RNA understanding tasks), ERNIE-RNA, Uni-RNA, UTR-LM, CaLM, RNAGenesis, GenerRNA, gRNAde (ICLR 2025 spotlight, 3D inverse design).
- **Protein**: AlphaFold 3 (*Nature* 2024), Boltz-1/Boltz-2 (MIT + Recursion 2025), Chai-1, Protenix, HelixFold3, RoseTTAFold All-Atom / RoseTTAFoldNA, RFdiffusion / RFdiffusion3, FrameFlow, FoldFlow, ESM3, Chroma, ProteinMPNN.
- **Cell**: scGPT (33 M cells), Geneformer, scFoundation, CellFM (100 M cells), Universal Cell Embeddings (UCE), GeneCompass, State (Arc 2025), Stack (Arc 2026), Tahoe-100M, GEARS.

### Layer 4 — Applications

- **Therapeutic discovery**: target identification, hit-to-lead, protein–ligand affinity (Boltz-2 Affinity Module), de novo binder design (RFdiffusion3), allosteric and cryptic-pocket discovery.
- **Genomic medicine**: CRISPR off-target screening (SMEU-native), saturation mutational scans, base- and prime-editor optimization, variant pathogenicity at ClinVar / GWAS resolution.
- **mRNA / RNA therapeutics**: 5′ and 3′ UTR optimization, codon usage tuned to organism + tissue + tRNA pool (WCC-native), modified-nucleoside (Ψ, m¹Ψ, m⁵C) substitution, LNP composition design.
- **Cell-state engineering**: virtual-cell perturbation prediction (MMC-native), cell-type-specific pharmacogenomics, CAR-T and iPSC reprogramming design, lineage-aware therapeutics.
- **Diagnostics**: whole-proteome variant scanning, splice-altering variant interpretation, cell-type expression deconvolution from bulk samples.
- **Manufacturing**: bioprocess optimization, host strain engineering for biologics, AAV / LNP titer prediction.

### Layer 5 — Deployment

Hybrid cluster: Crick-1 for biology, Volder-1 for geometric, Banach-1 for Euclidean transformer bulk, on Ethernet-native scale-up (Maia 200 ATL-compatible). Customer deployment options: on-prem cluster, hyperscaler-hosted (AWS / GCP / Azure / Oracle), pharma-private regional, sovereign biomedical compute.

---

## Market Perspective

### TAM and segments

Biology AI compute is currently ~5–8% of frontier AI compute spend (2025) and tracking to 12–18% by 2028 as AlphaFold-class structure prediction, Evo-class genomic foundation models, and virtual-cell models absorb an increasing fraction of pharma R&D budgets and hyperscaler training cycles. The four addressable buyer segments:

1. **Pharma R&D** (top 20 + next-50 mid-pharma) — currently ~$250 B/yr global R&D spend with declining productivity (Eroom’s Law: real cost per FDA-approved drug doubles every ~9 years since 1950). AI-native biology compute is the principal mechanism that has been hypothesized to invert that trend. Per-pharma annual compute budget for biology AI: $50–500 M, growing 40–60% CAGR.
2. **Foundation-model labs and AI biotech** — Isomorphic Labs, Recursion, Insilico Medicine, Schrödinger, Generate Biomedicines, Cradle, Profluent, EvolutionaryScale, Iambic, Inceptive, Latent Labs, Chai Discovery. Compute-intensive by design; the AlphaFold 3 / Boltz / RFdiffusion training-class workload is the lifeblood. Per-lab annual compute: $20–200 M.
3. **Hyperscalers and cloud** — NVIDIA BioNeMo, AWS HealthOmics, Google Cloud Life Sciences, Azure Genomics, Oracle Cerner. Compute supply, increasingly co-developed with chip and software vendors. The ecosystem-level move is to become the **default biology backend** of one or more hyperscalers.
4. **Sovereign biomedical compute and academic** — NIH / NHGRI, EMBL-EBI, Wellcome Sanger, RIKEN, BGI, KAUST. Lower per-account spend, higher policy leverage, key to dataset access and clinical validation.

### Value chain

Six links, with the economic rent migrating up the stack as the industry matures:

1. Process node (TSMC N2 / N2P / N2X) — commoditizing, ~3% of substrate-era value.
2. Packaging and HBM (CoWoS-L, HBM3e/HBM4) — bottlenecked, ~15%.
3. Compute die (Crick-1, Volder-1, Banach-1, GB300, TPU v7, Trainium3, WSE-3) — ~25%.
4. Compiler and software stack (CRICK-IR, BioNeMo, JAX/PyTorch ecosystem) — ~20%, rising.
5. Foundation models (AlphaFold 3, Boltz-2, Evo 2, RiNALMo, AIDO.RNA, State, Stack) — ~15%, rising fastest.
6. Applications and data (clinical workflows, pharma pipelines, diagnostics deployments) — ~22%, the eventual rent capture.

CRICK’s positioning: own the substrate (3, partial 2), the compiler (4), the open-source software (5), and partner up the value chain for 6. This is the **NVIDIA × CUDA × Hugging Face combined** pattern adapted to biology.

### Competitive map

The eight comparable substrates on biology workloads:

| Substrate | Primary primitive | Biology-effective $/TFLOPS-eq-W | Strategic posture |
|---|---|---|---|
| **Crick-1** | Pair-tensor iteration | **~$0.18** | Biology-native, full stack |
| Volder-1 | CORDIC rotation | ~$0.32 | Geometric / hyperbolic |
| Banach-1 | Fixed-point iteration | ~$0.50 | Convergent transformers |
| NVIDIA GB300 | Matmul | ~$0.78 | General-purpose dominance |
| AWS Trainium3 | Matmul | ~$0.35 | Cloud-captive training |
| TPU v7 Ironwood | Matmul | ~$0.42 | Inference-optimized |
| Cerebras WSE-3 | Matmul (wafer) | ~$0.61 | Long-context niche |
| Microsoft Maia 200 | Matmul | ~$0.55 | Azure-captive |

NVIDIA owns ~85% of the current frontier biology compute install base via raw GPU dominance plus the BioNeMo platform; that share is the principal asset under attack. The competitive question is not whether biology AI silicon emerges (Cerebras, Groq, SambaNova, Tenstorrent, Etched, and the hyperscaler ASICs all attest to demand), but whether the **substrate-algorithm co-design** thesis is correctly architected — and the pair tensor + SE(3) + diffusion + Hyena + Sherman-Morrison identification is the bet.

### Unit economics

Three before/after numbers that anchor the ROI calculation:

| Workload | Matmul-era (GB300 cluster) | Crick-1 cluster | Ratio |
|---|---|---|---|
| AlphaFold 3 frontier training run | $40–60 M | $6–10 M | 6× |
| Saturation mutational scan (300-residue protein) | ~$40 / scan, ~10 min wall-clock | ~$0.50 / scan, ~14 min single chip | 80× |
| 16K-chip cluster power for equivalent biology throughput | ~60 MW | 11.5 MW | 5× |
| Whole-genome CRISPR off-target screen (10⁹ guides) | ~$2.4 M / run | ~$30K / run | 80× |
| Cell-type-specific gene expression at population scale (10⁶ individuals × 50 tissues) | infeasible | ~$200 M one-time | unlocks |

The third row is the strategic differentiator. The first row sells to the AI-biotech and pharma-R&D segments. The second and fourth rows unlock workloads that don’t currently exist as commercial products. The fifth row is the population-genomic-medicine business line that no current substrate enables.

### Moats

Five compounding advantages, each with a defined time-to-replicate.

1. **Substrate-algorithm co-design** — the pair tensor / SE(3) / diffusion / Hyena / Sherman-Morrison identification matched to dedicated hardware blocks. Time to replicate by NVIDIA Rubin-class via 3D systolic-array extensions: ~3–4 years from architectural decision. Net moat window: 2026–2029.
2. **Compiler / IR** — CRICK-IR’s pair-tensor and SE(3) frame as first-class IR constructs; the col(F)/ker(F) tensor-attribute routing pattern. CUDA-equivalent ecosystem inertia compounds with each year of biology-AI developer adoption.
3. **Open-source software ecosystem** — CRICK-RNA / DNA / Protein / Cell as canonical, reproducible, benchmarked deployment of every frontier biology model with first-class Crick-1 backends. The Hugging Face × PyTorch × Linux pattern: dominance via developer surplus, not closed IP.
4. **Data and benchmark gravity** — partnership commitments with RNAcentral (release 26, 45 M sequences, 52 expert databases), Rfam, PDB, AlphaFold Protein Structure Database, Tahoe-100M, ENCODE, Human Cell Atlas. The benchmark-and-leaderboard control point that determines which model wins which capability claim.
5. **Pharma and clinical deployment partnerships** — co-development with top-5 pharma, top-5 AI biotech, and 2–3 hyperscalers; design wins in 3–5 sovereign biomedical compute projects (NHS, EU-1+ Million Genomes, Japan AMED, KAUST, NIH All of Us). Each partnership is a 5–10-year commitment with single-digit-percent compute share defended by switching cost.

### Adoption curve

- **2026 Q3** — Crick-1 first-silicon tape-out; CRICK-IR α; CRICK-RNA β; design-win count target 3.
- **2027 H1** — Volume production; CRICK-IR 1.0; full CRICK-RNA / DNA / Protein / Cell open-source releases; design-win count target 12.
- **2028** — Matmul-era biology compute hits the economic ceiling (P7). 30–40% of frontier biology training runs on Crick-1-class silicon. First FDA-cleared diagnostic incorporating whole-proteome saturation scanning (P2). Pharma R&D productivity inflection visible in pipeline KPIs.
- **2029** — Cell-type-specific pharmacogenomics commercial deployment (P3). Hybrid clusters (Crick + Volder + Banach + general GPU) become the standard biology compute pattern.
- **2030** — Integrated central-dogma compiler target as the default biology AI deployment (P5). Crick-2 successor in development with non-coding-RNA-regulation and 3D-chromatin-architecture caches.

---

## The Modules

Each module is open-source software with a defined Crick-1 backend, a defined PyTorch / JAX / Triton fallback, a defined hyperscaler-cloud reference deployment, and a defined pharma-partner adoption pipeline.

### CRICK-DNA — long-context genomic foundation modeling

Hosts Evo 2 (40 B params, 1 M-token context, 9.3 T training nucleotides spanning bacteria, archaea, eukaryotes, viruses; *Nature* 2026), Evo, HyenaDNA, Nucleotide Transformer, Caduceus, Borzoi, Enformer, CodonFM (Arc + NVIDIA, 2026), DNABERT-2. Native HFE backend; Mode J operating mode. Workloads: promoter and enhancer identification, transcription-factor binding site prediction, splice site prediction, variant pathogenicity, whole-genome de novo design, regulatory grammar discovery, CRISPR guide design, host-strain engineering.

### CRICK-RNA — sequence, structure, modification, design

Hosts the canonical RNA stack:
- **Foundation models** — RNA-FM, RiNALMo (650 M), AIDO.RNA (1.6 B), ERNIE-RNA, Uni-RNA, UTR-LM, CaLM, RNAGenesis, GenerRNA, RNAErnie.
- **Secondary structure** — RNAfold, Mfold, LinearFold / LinearPartition, CONTRAfold, EternaFold, MXfold2, UFold, SPOT-RNA / SPOT-RNA2, DSRNAFold, Sincfold; pseudoknot-aware and probing-conditioned variants.
- **Tertiary structure** — RhoFold+, NuFold, DRfold / DRfold2, trRosettaRNA, RoseTTAFoldNA, AlphaFold 3 / Boltz-2 / Chai-1 / Protenix / HelixFold3 (license-permitting), with optional FARFAR2 / SimRNA refinement.
- **Inverse design** — gRNAde (SE(3)-equivariant 3D inverse design, ICLR 2025 spotlight, ribozyme-validated), RDesign, NA-MPNN, RNAGenesis design head, structure-to-sequence aptamer pipelines.
- **Therapeutic RNA** — Optimus 5-Prime, UTailoR (*iScience* 2025), Saluki for half-life, CodonBERT / CaLM / CodonFM for ORF optimization, modified-nucleoside (Ψ, m¹Ψ, m⁵C) substitution, LNP-composition AI (D-MPNN over ionizable-lipid SMILES, protein-corona prediction with SHAP attribution).
- **Splicing** — SpliceAI (*Cell* 2019), Pangolin (tissue-specific), SpliceTransformer (*Nat Commun* 2024), AbSplice, CADD-Splice, MMSplice / MTSplice.
- **Epitranscriptome** — CHEUI for joint m⁶A + m⁵C from nanopore signal, m6ATM, m6Anet, EpiNano; DRACH-motif and writer/eraser/reader stack.
- **circRNA** — ORNA, CirPure / CirPrecise, Clean-PIE engineered circularization; circRNA-vaccine and decoy-circRNA pipelines.
- **CRISPR-Cas13** — RfxCas13d / CasRx, PspCas13b, hfCas13d / hfCas13X (high-fidelity), TIGER and DeepCas13 guide-design models, circular gRNA biostability.
- **RNA–protein interactions** — ZHMolGraph, ZeRPI, GraphProt2, DeepCLIP, RBPNet, eCLIP / iCLIP / PAR-CLIP integration.
- **RNA-targeting small molecules** — SMARTBind, RNAmigos2, RNAsmol, riboswitch-pocket screening, RIBOTAC scaffolds.
- **Non-coding RNA** — lncRNA coding-potential and subcellular-localization, miRNA target prediction, piRNA / snoRNA / tRNA / snRNA tooling; **RNAcentral release 26** integration (45 M sequences, 52 expert databases, gene-level entries across 204 organisms).

### CRICK-Protein — structure, complex, design

AlphaFold 3 / Boltz-1 / Boltz-2 (Affinity Module) / Chai-1 / Protenix / HelixFold3 / RoseTTAFold All-Atom for prediction; RFdiffusion / RFdiffusion3 / FrameFlow / FoldFlow / Chroma for design; ESM3 for evolutionary-scale generative protein modeling; ProteinMPNN and LigandMPNN for sequence design; PocketGen and ESM3-fold-conditioning for binder design; AlphaMissense and EVE for variant pathogenicity. Native TAE + SFB + DDU backend; Modes H + I.

### CRICK-Cell — virtual cells and perturbation

scGPT (33 M cells), Geneformer, scFoundation, CellFM (100 M cells), Universal Cell Embeddings (UCE), GeneCompass, RegFormer, State (Arc 2025), Stack (Arc 2026), GEARS for graph-perturbation, RNA-velocity stack (scVelo, veloVI, VeloVAE, Dynamo, DeepVelo, RegVelo, cell2fate, PRESCIENT). Spatial transcriptomics (Visium / Xenium / MERSCOPE / Stereo-seq) and multi-omics integration. Native MMC backend for cell-type-specific epigenetic conditioning; cross-modal Mode H + Mode J operation.

### CRICK-1 — silicon

128 PTIUs on TSMC N2P; 220 B transistors; 873 MB SRAM; 384 GB HBM4 @ 12.8 TB/s; 4.0 TB/s Ethernet scale-up; 700 W TDP; full hardware-block manifest (TAE / SFB / DDU / HFE / SMEU / WCC / MMC / AAB / CM) detailed in the Crick-1 spec.

### CRICK-IR — compiler

MLIR dialect; PyTorch and JAX front-ends; Triton-compatible kernel emission; explicit pair-tensor and SE(3) frame IR types; col(F)/ker(F) tensor attributes; central-dogma pipeline lowering; ten operating modes (seven inherited from Banach-1 / Volder-1 plus H, I, J).

---

## Notes

A small number of technical and economic identifications that constrain the design and the business.

**N1.** The pair tensor is the universal intermediate of biology compute. Every modern structure-prediction architecture (AlphaFold 2/3, Boltz-1/2, Chai-1, Protenix, HelixFold3, RoseTTAFoldNA, RhoFold+) evolves it; every modern protein design architecture (RFdiffusion, RFdiffusion3, FrameFlow, FoldFlow, Chroma) refines noised versions of it; every modern virtual-cell model implicitly conditions cell-state representations on it via the embedded structure prior. Substrate identification: the cache, the systolic dataflow, and the IR type must be sized and shaped for the pair tensor.

**N2.** Triangle inequality is the geometric backbone. Any predicted 3D structure must satisfy the triangle inequality on its pair-distance matrix. Triangle attention + triangle multiplication is the hardware enforcement; substrates that emulate this via general matmul pay a 4–6× compute penalty on every Pairformer block.

**N3.** SE(3) equivariance is exact at hardware level on Crick-1, approximate at software level on matmul-only substrates. The polynomial Taylor-series approximation of the rotation matrix on a Vector Processing Unit drifts by 10⁻⁴ to 10⁻⁶ per rotation; the CORDIC cascade is exact to the chosen iteration count.

**N4.** Diffusion sampling is fixed-point iteration. Each denoising step is a single Banach contraction toward the score-determined direction; 200–500 steps for AlphaFold 3, RFdiffusion3, and the SE(3)-flow-matching family. Anderson acceleration applies; the Contraction Monitor reports convergence; the DDU handles noise scheduling natively.

**N5.** Long-context Hyena is FFT-native, and FFT is a CORDIC operation by construction. The Hyena Filter Engine implements the long convolution as a butterfly-structured CORDIC cascade; 1-megabase context on a single chip; longer contexts chunk over the Ethernet fabric.

**N6.** CRISPR is the Sherman-Morrison update on the genomic Fisher matrix. Saturation mutational scanning, base-editing optimization, off-target screening, allele-specific expression prediction, ClinVar variant annotation, and fine-resolution GWAS are all rank-one-update workloads. SMEU compresses 5,700-variant scans from ~47.5 GPU-hours to ~14 single-chip-minutes; the 80–100× ratio is the principal CRISPR-screening unit-economics driver.

**N7.** The 64→20 surjection is the genome’s native error-correcting code, with 44 of 64 codons synonymous. The 6.0-bit-per-codon nominal information content reduces to ~4.32 bits of amino-acid-functional content; the remaining 28% is regulatory ker(F) (codon usage bias, tRNA-pool compatibility, mRNA folding constraints, immunogenicity profile). The Wobble Code Cache enables runtime-selectable col(F)-only inference (28% bandwidth saving) or full-codon inference for mRNA design.

**N8.** The epigenome is heritable ker(F). Two cells with identical DNA (liver vs. neuron) differ in epigenetic configuration — methylation, histone marks, chromatin accessibility, 3D contact frequencies. Cell-type-specific genomic medicine is a joint (genome, epigenome) tensor workload; the Methylation Marker Cache holds the ker(F) on-chip with the col(F) accessible in HBM. No matmul-only substrate has the architectural separation; the MMC enables 10–50× the cell-type × individual × tissue throughput on cell-type-specific pharmacogenomics.

**N9.** Hardware lottery applies in both directions (Hooker, 2020). The matmul-era’s dominance is the path-dependent outcome of GEMM-optimized substrates winning the 2012–2025 decade; the next decade’s biology compute is a co-design opportunity. The risk symmetry: if NVIDIA’s Rubin or post-Rubin generations add 3D systolic-array extensions (the most natural matmul-side response to the pair-tensor identification), the workload-effective gap compresses. Crick-1’s design assumes 3–4 years of architectural lead time.

**N10.** Eroom’s Law inversion is the strategic prize. Pharma R&D cost per FDA-approved drug has doubled every ~9 years for 70 years; AI-native biology compute is the principal mechanism with the structural capacity to invert that trend. The economic upper bound on biology AI compute spend is set by the ratio of pharma R&D productivity recovery to the marginal cost of compute; both numerators are large and both denominators are falling fast. Plausible 2030 biology AI compute TAM: $40–80 B/yr, with ecosystem-level rent capture (substrate + compiler + open-source models + clinical deployment partnerships) reaching ~20–30% of that.

**N11.** Open-source is the rational software posture. The closed-weights model (Anthropic / OpenAI / Google DeepMind / Isomorphic) is correct for general-purpose AI but wrong for biology AI: data provenance, reproducibility, regulatory transparency, and academic-clinical adoption all favor open weights and open IR. CRICK-RNA / DNA / Protein / Cell are MIT-licensed; the moat is substrate + compiler + ecosystem, not model weights.

**N12.** The companion theoretical framework is load-bearing. Each chip and each module rests on a documented mathematical identification — Dirac Representation Hypothesis, Bregman Closure, Score Closure, Geometric Descent, Minkowski Representation Theory, Closed Future Cone, Operators Without Opponents, and the CAJAL feature-circuit-attribution architecture for mechanistic interpretability. The set is the substrate-algorithm-software co-design’s intellectual foundation.

---

## Predictions

Eight wagers, each testable, each priced.

**P1.** Pair-tensor-native silicon dominates biology workloads. 4–12× per-chip advantage on AlphaFold 3 / Boltz-2 / RFdiffusion3 / Evo 2 inference and training produces 3–5× cost reduction for pharma and AI-biotech compute pipelines by 2028. Top-5 AI-biotech labs adopt within 24 months of Crick-1 availability.

**P2.** The Sherman-Morrison Edit Unit redefines CRISPR computational screening economics. Saturation mutational scanning moves from $40/scan to $0.50/scan. First FDA-cleared diagnostic incorporating whole-proteome saturation scanning approved by 2028.

**P3.** The Methylation Marker Cache enables cell-type-specific genomic medicine at population scale. First commercial cell-type-specific pharmacogenomics product deployed by 2029.

**P4.** The col(F)/ker(F) architectural separation becomes the standard pattern for biology compute. Future biology accelerators inherit the cache-level separation and extend it to additional ker(F) data structures (non-coding RNA regulation, 3D chromatin architecture, lineage-specific developmental marks).

**P5.** The central-dogma compute pipeline becomes a unified compiler target. The current separation between genomic, RNA, structure, and virtual-cell foundation models is replaced by integrated CRICK-IR programs by 2030.

**P6.** Hybrid clusters become the standard biology compute deployment. Crick-1 for biology, Volder-1 for geometric / hyperbolic, Banach-1 for the Euclidean transformer bulk, all on Ethernet-native scale-up.

**P7.** The matmul-era biology compute substrate hits an economic ceiling by 2028. Pair tensor is not a matrix product; SE(3) equivariance is not a Euclidean operation; diffusion is not a single-pass forward; Hyena is not quadratic attention; Sherman-Morrison is not a full inference. The 3–12× emulation overhead becomes prohibitive at population-scale workload volume.

**P8.** Eroom’s Law inverts in 2027–2029, attributable in the causal-attribution literature to AI-native biology compute. CRICK ecosystem captures 15–25% of the biology AI compute layer’s rent.

The wagers fail diagnostically in two ways. First, if biology AI workload growth slows (P1, P2, P3 fail), Crick-1’s specialization is wasted and the chip reduces to a competent matmul accelerator at slightly-lower-than-peer FLOPs. Second, if matmul-era substrates ship efficient pair-tensor and triangle-attention extensions (NVIDIA Rubin or post-Rubin 3D systolic, TPU v8/v9 XLA custom kernels), the workload-effective gap compresses and the moat window shortens from 2026–2029 to 2026–2027. Both failure modes are publicly observable.

---

## Sources

Watson & Crick 1953 · Crick 1958 / 1966 / 1970 · Anfinsen 1973 · Doudna & Charpentier 2014 · Komor 2016 · Anzalone 2019 · Banach 1922 · Sherman & Morrison 1950 · Volder 1959 · Walther 1971 · Anderson 1965 · Hooker 2020 · Dao 2022 (FlashAttention) · Poli 2023 (Hyena) · Gu & Dao 2023 (Mamba) · Jumper 2021 (AlphaFold 2) · Abramson 2024 (AlphaFold 3) · Wohlwend 2024 (Boltz-1) · Boltz-2 2025 · Chai-1 2024 · Protenix 2025 · HelixFold3 2024 · Watson 2023 (RFdiffusion) · Butcher 2025 (RFdiffusion3) · Yim 2023 (FrameFlow) · FoldFlow 2023 · Hayes 2024 (ESM3) · Ingraham 2023 (Chroma) · Dauparas 2022 (ProteinMPNN) · Chen 2024 (RNA-FM) · Shen 2024 (RhoFold+) · Penić 2025 (RiNALMo) · AIDO.RNA 2024 · ERNIE-RNA 2024 · Joshi & Liò 2025 (gRNAde) · RNAGenesis 2024 · Jaganathan 2019 (SpliceAI) · Zeng & Li 2022 (Pangolin) · SpliceTransformer 2024 · CHEUI 2024 · m6ATM 2024 · UTailoR 2025 · Optimus 5-Prime · Saluki · Nguyen 2024 (Evo) · Brixi 2026 (Evo 2) · CodonFM 2026 · Cui 2024 (scGPT) · Theodoris 2023 (Geneformer) · CellFM 2025 · UCE · GeneCompass · State 2025 · Stack 2026 · Tahoe-100M · Bunne 2024 (Towards a Virtual Cell) · RNAcentral release 26 (2026) · Ren — CRICK / Banach-1 / Volder-1 / CAJAL / Dirac Representation Hypothesis / Bregman Closure / Score Closure / Geometric Descent / Minkowski Representation Theory / Closed Future Cone / Operators Without Opponents / Poincaré Is All You Need (2026).

---

## Closing

The data was sequenced. The structure was extracted. The function was predicted. The cell was simulated. The substrate was matmul. The substrate becomes the pair tensor, the rotation, and the fixed point. The chip is the central dogma in silicon. The compiler is the central dogma in code. The software is the central dogma in open source. The strategy is the central dogma in capital.

**CRICK is the ecosystem.**
