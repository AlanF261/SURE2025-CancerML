# SURE2025-Cancer-ML

Code primarily based off code from Nucleotide Transformer v2, DNABERT-2, and MethylBERT. 

To train on your own data: 

Using Bismark, create a reference genome, align your fastq files, sort your fastq files, and index them. 

Create .txt documents containing lists of filepaths for each of the following cohorts: discovery (for pretraining), healthy discovery, cancer discovery, healthy validation, and discovery validation. 

Provide filepaths of these .txt documents in either pretrian.py or finetune.py.
