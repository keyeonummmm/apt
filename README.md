# APT: Artificial Intelligence Visualization Project

## Overview

This project is part of my thesis work in the Drawing & Painting program at OCAD University. The thesis explores the conceptual differences and similarities between human and artificial intelligence. APT serves as a visual illustration of AI "under the hood," providing an artistic interpretation of how artificial intelligence functions.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/keyeonummmm/apt.git
cd apt
```

## Usage

There are two ways to view the visualization:

### GUI Interface
```bash
cd finetuning
python apt_viewer.py
```

### Terminal Interface
```bash
cd finetuning
python sample.py
```

## Fine-tuning Your Own GPT-2 Model

To fine-tune GPT-2 with my settings:

1. Create your data file in the root directory
2. Prepare your data:
   ```bash
   python data-prep.py
   ```
3. Start the fine-tuning process:
   ```bash
   python finetuning.py
   ```

## Acknowledgments

Credit to Andrej Karpathy for his nanoGPT project and the valuable knowledge he shared in his videos to the public, helping us understand LLMs in general with a better sense of what, how, and why they work.

Resources:
- [Karpathy's nanoGPT project](https://github.com/karpathy/nanoGPT)
- [Karpathy's YouTube channel](https://www.youtube.com/c/AndrejKarpathy)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
