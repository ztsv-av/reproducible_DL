This repo is a final project for the **Building Reproducible Analytical Pipelines** course taught at the University of Luxembourg by Prof. Dr. Bruno Rodrigues. Author: Anton Zaitsev.

The repo is designed to easily reproduce training &amp; evaluation of a ML model on the MNIST dataset.

# Steps Done

1. Git:
   1. Created github repo.
   2. Cloned repo.
2. Virtual environment:
   1. Created: `python3 -m venv venv`
   2. Activated: `venv\Scripts\activate`
   3. Installed dependencies: `pip install torch torchvision`
   4. Freeze dependencies: `pip freeze > requirements.txt`

The idea is to always produce the same training and evaluation results (model accuracy) as the ones initially produced. For example, imagine someone wrote a paper on a machine learning algorithm and got some results. Natually, everyone wants to see to code used and reproduce the same results as the authors to make sure that the results are valid. In this repo we build a pipeline to classify MNIST images. For reproducibility to hold, the following criterias should be always the same:
1. PyTorch seed.
1. Training and testing data.
2. Data preprocessing (in our case we do standardization, so we should always have the same computed mean and standart deviation).
3. Model architecture.
4. Same loss function and optimizer (in our case we hardcode them during training procedure).