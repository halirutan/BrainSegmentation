# Installation

This guide will help you set up the Brain Segmentation project on your local machine.

## Prerequisites

- Python 3.7 or higher
- Conda (recommended for managing environments)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/BrainSegmentation.git
cd BrainSegmentation
```

### 2. Set Up the Environment

We provide a Conda environment file to make setup easier:

```bash
conda env create -f python_setup/environment.yml
conda activate brainseg
```

### 3. Verify Installation

To verify that the installation was successful, you can run a simple test:

```bash
python -c "import brainseg; print('Installation successful!')"
```

## Troubleshooting

If you encounter any issues during installation, please check the following:

- Ensure you have the correct version of Python installed
- Make sure Conda is properly installed and in your PATH
- Check that all dependencies were installed correctly

If problems persist, please [open an issue](https://github.com/yourusername/BrainSegmentation/issues) on our GitHub repository.