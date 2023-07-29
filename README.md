# Watermarking ML/DL Models

This repository aims to address the challenges of protecting intellectual property in the context of Deep Neural Networks (DNNs) by implementing the techniques described in the paper titled "Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring."

## Background

We utilized the code published by the original authors, which can be found on [GitHub](https://github.com/adiyoss/WatermarkNN). For our specific use-case, we made some minor modifications to adapt the code:

- Replaced params with config files to improve usability.
- Incorporated a progress bar using tqdm to provide better visibility during the execution.
- Expanded the model options by adding Densenet.
- Included the MNIST dataset to enhance the range of supported datasets.

These modifications were primarily aimed at tailoring the implementation to our specific requirements. Notably, we introduced custom transformations for the MNIST dataset and made adjustments to the Densenet architecture to allow for re-initialization of the classification layer at the end.

Furthermore, we addressed an issue in the fine_tuning.py script, where the evaluation on the trigger set was performed using the original classification layer instead of the fine-tuned one. We fixed this inconsistency to ensure accurate evaluation results.

## Usage

To simplify the usage of our implementation, we have provided Jupyter Notebooks that can seamlessly run in the Google Colab environment. By uploading any of the two provided IPython notebook files to Colab, you can effortlessly reproduce our experiments. These notebooks automatically handle all the dependencies, ensuring a smooth execution.

Please feel free to explore the code and experiment with watermarking techniques to safeguard your Deep Neural Networks. If you have any questions or feedback, don't hesitate to contact us.

Note: It's always advisable to respect the terms and conditions set by the original authors when using their code or techniques.
