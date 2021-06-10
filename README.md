Welcome to the repository for "BasisDeVAE: Interpretable Simultaneous Dimensionality Reduction and Feature-Level Clustering with Derivative-Based Variational Autoencoders".

The code provided here builds on the original implementation of BasisVAE by Kaspar
Märtens in PyTorch, to be found at https://github.com/kasparmartens/BasisVAE.

The files `decoder.py`, `encoder.py`, `helpers.py` and `VAE.py` contain the core
functionality of the VAE, DeVAE, BasisVAE and BasisDeVAE frameworks.

The file `main.py` demonstrates the method by i) executing `synth_data_gen.py`
to generate synthetic data and ii) fitting BasisDeVAE to this data as done in the paper.

Core dependencies (excluding PyTorch GPU, which should be configured separately ensuring
compatible CUDA support and device drivers) are contained within `requirements.txt` and
can be installed via `pip install -r requirements.txt`.
