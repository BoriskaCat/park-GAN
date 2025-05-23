# cGAN for Park Masterplan Generation

This repository contains a cGAN-based pipeline for generating schematic park masterplans conditioned on a contour image and scalar area value.

## 📂 Structure

- `train.ipynb` – Main training notebook (initial 1200 epochs).
- `fine_tuning.ipynb` – Fine-tuning phase (additional 300 epochs).
- `test.py` – Loads the trained model and generates a masterplan given a contour image and target area.

## 🚀 Usage

### Training

Run the training notebooks in order:

1. `train.ipynb` – trains the base model  
2. `fine_tuning.ipynb` – performs fine-tuning for improved realism and area response

> Training logs and model weights will be saved to the predefined directories in each notebook.  
> **Note:** Don’t forget to adjust `w_p` and `w_fm` after the 1350th epoch!

### Testing

To generate a plan from a custom contour, run:

```bash
python test.py --image path/to/contour.jpg --area 3.25
