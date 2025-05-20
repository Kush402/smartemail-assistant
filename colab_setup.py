import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SmartEmail Assistant - Google Colab Setup\n",
                "\n",
                "This notebook helps you set up and run the SmartEmail Assistant project on Google Colab with GPU acceleration."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Install Required Libraries"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "!pip install -q torch transformers peft datasets bitsandbytes accelerate wandb"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Clone the Repository"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "!git clone https://github.com/Kush402/smartemail-assistant.git\n",
                "%cd smartemail-assistant"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Create Project Structure"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "!mkdir -p data/raw data/processed model/checkpoints model/peft_adapter"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Upload Training Data\n",
                "\n",
                "Upload your `raw_emails.csv` file to the `data/raw` directory. You can do this by:\n",
                "1. Click on the folder icon in the left sidebar\n",
                "2. Navigate to `data/raw`\n",
                "3. Click the upload button and select your file"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Prepare the Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "!cd data && python prepare_data.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Train the Model\n",
                "\n",
                "This will take some time. The model will be saved to the `model` directory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "!cd fine-tune && python trainer.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Test the Model\n",
                "\n",
                "After training is complete, you can test the model with some example inputs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "from app.main import load_model_and_tokenizer, generate_response\n",
                "\n",
                "# Load the model\n",
                "model, tokenizer = load_model_and_tokenizer()\n",
                "\n",
                "# Test with an example\n",
                "subject = \"Leave Request\"\n",
                "email_body = \"I need to take leave from 15th to 20th June for personal reasons.\"\n",
                "\n",
                "response = generate_response(model, tokenizer, subject, email_body)\n",
                "print(\"Generated Response:\")\n",
                "print(\"-\" * 80)\n",
                "print(response)\n",
                "print(\"-\" * 80)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Save the Model\n",
                "\n",
                "After training, you can download the model files to your local machine:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "from google.colab import files\n",
                "\n",
                "# Create a zip file of the model directory\n",
                "!zip -r model.zip model/\n",
                "\n",
                "# Download the zip file\n",
                "files.download('model.zip')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Additional Notes\n",
                "\n",
                "1. Make sure to select a GPU runtime in Colab (Runtime > Change runtime type > GPU)\n",
                "2. The training process might take 1-2 hours depending on the dataset size\n",
                "3. You can monitor the training progress through the output\n",
                "4. The model checkpoints will be saved in the `model/checkpoints` directory\n",
                "5. The LoRA adapter will be saved in the `model/peft_adapter` directory\n",
                "\n",
                "### Troubleshooting\n",
                "\n",
                "If you encounter any issues:\n",
                "1. Make sure you're using a GPU runtime\n",
                "2. Check if all dependencies are installed correctly\n",
                "3. Verify that your training data is in the correct format\n",
                "4. Ensure you have enough disk space in Colab"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebook
with open('colab_setup.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 