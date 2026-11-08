ICESat-2 Crevasse Detection and Depth Retrieval

This repository provides the source code for an automated framework for crevasse detection and depth estimation using ICESat-2 ATL03 photon data over the Greenland Ice Sheet (GrIS).

Overview

The proposed framework consists of three main processing modules:

Roughness-based Candidate Region Screening (01_roughness_screening.py)

Identifies potential crevassed regions using a roughness-based screening approach to reduce the computational cost of subsequent processing.

Photon Denoising (02_photon_denoising.py)

Extracts glacier surface signal photons using direction-adaptive similarity-weighted density estimation and local spatial-continuity filtering.

Crevasse Extraction and Depth Estimation (03_crevasse_extraction.py)

Reconstructs the glacier surface, extracts individual crevasses, and estimates their geometric characteristics and depths.

Processing Workflow

The three modules should be executed sequentially:

Roughness-based screening → Photon denoising → Crevasse extraction and depth estimation.

The output of each module serves as the input to the subsequent processing step.

Repository Structure

ICESat2-Crevasse-Detection-and-Depth-Retrieval/
│
├── 01_roughness_screening.py
├── 02_photon_denoising.py
├── 03_crevasse_extraction.py
├── README.md
└── .gitignore
