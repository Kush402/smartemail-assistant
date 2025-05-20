# SmartEmail Assistant

A professional email response generator using fine-tuned GPT-2 model with LoRA (Low-Rank Adaptation).

## Project Structure

```
smartemail-assistant/
├── app/                    # Application code
│   ├── main.py            # Main application script
│   ├── utils.py           # Utility functions
│   └── prompts.py         # Prompt templates
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data files
│   └── prepare_data.py   # Data preparation script
├── fine-tune/            # Model fine-tuning code
│   ├── trainer.py        # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── dataset_loader.py # Dataset loading utilities
│   └── config.yaml       # Training configuration
├── model/                # Model files
│   ├── checkpoints/     # Model checkpoints
│   ├── peft_adapter/   # LoRA adapter files
│   └── tokenizer/      # Tokenizer files
└── notebooks/           # Jupyter notebooks
    └── eda_preprocessing.ipynb  # Data analysis notebook
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Kush402/smartemail-assistant.git
cd smartemail-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the data:
```bash
cd data
python prepare_data.py
```

2. Train the model:
```bash
cd fine-tune
python trainer.py
```

3. Run the application:
```bash
cd app
python main.py
```

## Features

- Fine-tuned GPT-2 model using LoRA for efficient training
- Professional email response generation
- Interactive command-line interface
- Customizable training parameters

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- Datasets
- Pandas

## License

MIT License 