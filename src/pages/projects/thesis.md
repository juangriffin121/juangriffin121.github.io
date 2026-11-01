---
layout: ../../layouts/MarkdownLayout.astro
title: "Short Linear Motifs in Amyloidogenic and Immune-Response Proteins"
description: "Bachelor's thesis in Biotechnology and Molecular Biology"
thumbnail: '/thumbs/thesis.png'
tags: ["Bioinformatics", "Python"]
---

**Bachelor's thesis (Licenciatura en Biotecnología y Biología Molecular), Universidad Nacional de La Plata**
Worked in the Structural Bioinformatics Group, Universidad Nacional de Quilmes.
Advisor: Nicolás Palopoli · Co-advisor: Gustavo Parisi · Defended: 12 September 2025

**Code and data:** [github.com/juangriffin121/Tesis](https://github.com/juangriffin121/Tesis)

## Summary

Short linear motifs (SLiMs) are 3–15 residue sequence patterns, usually found in intrinsically disordered regions, that let proteins bind each other. Mutations in them are linked to roughly 20% of human disease-causing mutations in this class of region, and pathogens often mimic them to hijack host cells. They are hard to predict because their short length makes sequence patterns match by chance.

This project used computational methods to study SLiMs in two settings where they matter: **amyloid-forming proteins** and the **innate immune response**. It ended with a new motif class, curated by me and deposited in the [ELM database](http://elm.eu.org), the main reference resource for SLiMs.

## What I did

### 1. Do SLiMs and amyloid-prone regions co-localize?

- Cross-referenced **AmyPro** (81 validated amyloid proteins) with **3,953 annotated ELM instances**, giving 10 usable proteins (p53, APP, Tau, tubulin, vitronectin, androgen receptor, nucleophosmin, acetylcholinesterase, PrP, α-synuclein) with 37 SLiM instances in 21 classes.
- Measured overlap and the minimum distance from each SLiM to the nearest amyloid-prone region, normalized by protein length.
- Tested against a null model: **10,000 random re-placements** of SLiMs along each protein, with amyloid regions held fixed.
- **Result:** SLiMs are clearly biased toward disordered regions, but their position relative to amyloid regions is mostly consistent with chance. Only the 5–15% distance range stood out. This is a small dataset, so the result argues for further study rather than a firm conclusion.

![alt text](/thesis/slims.png) 

### 2. A candidate new motif in APP: YKFFE and AP4

- **APP** the precursor protein to the peptide whose amyloid accumulation is responsible for Alzheimer's disease had some evidence of SLiMs not covered in ELM.
- Found that the APP tail motif **YKFFE**, which binds the μ subunit of the AP4 adaptor complex at a non-canonical pocket, is not covered by any ELM class. I proposed the pattern `YX[FYL][FL]E` as a possible new TRG_ENDOCYTIC subclass, with **LAMP-2** as a candidate second instance.
- Ran a co-evolution analysis (BLAST, Clustal Omega, Python). Across 1,621 APP homologs the motif is 98.1% conserved on average, and the AP4M1 binding pocket keeps the chemical character of its residues even where the residues vary.
- The evidence was promising but not enough for a formal deposit, so the work stops at a well-supported candidate.

![alt text](/thesis/ykffe.png) 

### 3. A new curated ELM class: DOC_TBK1_STING_1

- Studied how STING recruits the kinase TBK1 to trigger the type I interferon response to cytosolic DNA.
- Built the evidence base: BLAST homologs, multiple alignments, a neighbor-joining tree, per-position conservation, sequence logos, disorder context from InterPro/MobiDB/AlphaFold, and crystal structures (PDB 6O8C, 6NT9) visualized in PyMOL.
- Derived the consensus pattern with SLiMMaker and searched the human proteome with SLiMSearch. The pattern is `[DE]XPX[PST]L[RKHNQ][ST][DN]`.
- The search surfaced **IRAK2** as a putative additional instance, in a disordered region of a protein that sits in the same signaling pathway.
- Curated **8 instances** (human, mouse, pig and chicken STING, plus IRAK2) and submitted them to the ELM team.
- **Outcome:** the class, **DOC_TBK1_STING_1**, was included in the 2024 ELM database update. This work was published in a paper[^1] on which I am a co-author.

![alt text](/thesis/sting.png) 

## Tools and methods

| Area | Tools |
|---|---|
| Language and libraries | Python, pandas, NumPy, Biopython, Matplotlib, seaborn, Jupyter |
| Sequence analysis | BLAST, Clustal Omega, Jalview, WebLogo, regular-expression pattern search |
| SLiM resources | ELM, SLiMSearch, SLiMMaker, articles.ELM |
| Protein and interaction databases | UniProt, AmyPro, MobiDB, InterPro, STRING, PDB |
| Structure | AlphaFold models, PyMOL (structural alignment and figures) |
| Statistics | Permutation-style null model (10,000 simulations), coverage and distance distributions |

## What I took from it

- Designing a null model so that a pattern in a small biological dataset is not mistaken for signal.
- Turning scattered literature and structural data into a structured, peer-reviewed database entry, which is biocuration.
- Combining sequence conservation, disorder prediction, structure and interaction networks to argue for or against a functional motif.
- Being honest about evidence: one candidate class was published, while another (YKFFE) was left as a candidate.

## Repository layout

```
Amylo dataset/ · Amylo_dataset_search/   AmyPro and MobiDB data, ELM instances
Get_slims/                                regex scan of ELM classes over homolog alignments
INTERACTIONS/                             STRING networks, APP / AP4M1 co-interactors
motif and pocket/ · coevolution/          APP and AP4M1 alignments, logos, trees
STING-TBK/                                STING alignment, tree, identity matrix, motif checks
```

[^1]: [Kumar, M. et al. ELM-the Eukaryotic Linear Motif resource-2024 update.](https://pubmed.ncbi.nlm.nih.gov/37962385/)
