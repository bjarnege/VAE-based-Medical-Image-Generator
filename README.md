# VAE-based-Medical-Image-Generator

Idea: Generate images from different imaging modalities by training on MedMNIST dataset
> Basic Task (0.3):
* Implement and train VAE to generate medical images from MedMNIST dataset

>Extension (0.7):
* Implement VAE variant for (a) disentanglement or (b) conditional training and evaluate for >= 3 different modalities (e.g. X-ray, Pathology, Dermatology)


## Dependencies

Run:

```bash
poetry config virtualenvs.in-project true
```

to install virtualenv in the project folder for better ide support

Then install all dependencies:

```bash
sudo apt-get install graphviz
poetry install
```

Start a shell session within

```bash
poetry shell
```

## Run program

TODO