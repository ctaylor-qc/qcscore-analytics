# QC Score Analytics

This repository provides customer-facing assets for running analytics on Promethium QC Score output.

# Purpose
Conventional docking workflows typically produce one score per ligand.
QC Score instead provides a residue-level interaction energy profile across the binding pocket.
The goal of this repository is to help users transform that richer output into practical lead-optimization insight.

This repo is intended to host:

* QC Score interaction analysis scripts
* Example config.json files
* Whitepaper PDF
* Demo CSV used in the whitepaper example

# What the analysis pipeline does
The QC Score Residue Interaction Analysis pipeline converts residue-resolved QC Score output into interpretable outputs that answer three core questions:

* Which residues best discriminate active vs less-active ligands?
* Are there distinct binding mode families in the ligand set?
* Which ligand pairs show large activity differences despite similar interaction profiles (activity cliffs)?

# Processing and analysis steps
* Optional multi-pose preprocessing: Boltzmann averaging collapses multiple poses into one representative row per ligand, using score+strain as the energy weight.
* Automated quality control: detects clashing poses, localized residue artifacts, high-strain outliers, and computational boundary residues.
* Residue ranking: applies partial least squares regression and Spearman rank correlation to identify residues that co-vary with binding affinity.
* Interaction-family discovery: uses k-means clustering with automatic cluster selection to identify binding-mode families.
* Activity-cliff detection: flags ligand pairs for high-resolution FSAPT follow-up.

# Outputs
Designed for users without deep computational chemistry expertise:

* Ranked residue tables
* Annotated bar charts
* Cluster visualizations
* Plain-language summary report with interpretation guidance

# Intended use
Customers can use the scripts and examples in this repository to run the analytics engine on QC Score outputs and generate interpretable, residue-level decision support for lead optimization.

# Dependencies
Both scripts require Python 3.8+ with the following packages:

pip install pandas numpy scipy scikit-learn matplotlib seaborn
