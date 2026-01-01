# Named-Entity-Recognition-RNN
# Recurrent Neural Network for Named Entity Recognition (NER)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat&logo=keras)

## Project Overview
This project implements a **Recurrent Neural Network (SimpleRNN)** for Named Entity Recognition (NER) tasks. The model is designed to process sequential text data and classify words into predefined entity categories (e.g., Person, Location, Organization, Time).

The project progresses from a basic prototype to a more robust model using a real-world dataset.
1.  **Prototype:** A simple RNN trained on dummy sentences to validate the pipeline.
2.  **Full Model:** An enhanced RNN trained on the Kaggle NER dataset with optimized hyperparameters.

## Model Architecture
The network is built using the Keras Sequential API with the following layers:
* **Embedding Layer:** Converts integer-encoded vocabulary into dense vectors of fixed size.
* **SimpleRNN Layer:** Processes the sequence of embeddings to capture temporal dependencies (50 units).
* **Dropout Layer:** Applied (0.1) to prevent overfitting during training.
* **TimeDistributed Dense Layer:** Outputs a probability distribution over the entity tags for each time step in the sequence.

##  Dataset & Preprocessing
* **Dataset:** The model uses the "ner.csv" dataset (sourced from Kaggle) containing thousands of labeled sentences.
* **Tokenization:** Text is tokenized and converted to sequences, handling out-of-vocabulary words with an `<OOV>` token.
* **Padding:** Sequences and labels are padded to a uniform length to allow for batch processing.
* **Label Encoding:** Entity tags (e.g., `B-geo`, `I-tim`) are encoded into unique integers using Scikit-learn's `LabelEncoder`.

##  Results
The model was trained for 50 epochs with a batch size of 32.
* **Validation Accuracy:** ~94%
* **Validation Loss:** ~0.43

The model successfully identifies entities in unseen test sentences:
* *Input:* "London is based in Paris"
* *Prediction:* **London** -> `B-geo` (Geographical Entity)

##  Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Language:** Python

##  How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/RNN-Named-Entity-Recognition.git](https://github.com/YOUR-USERNAME/RNN-Named-Entity-Recognition.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn
    ```
3.  **Load the Data:**
    Ensure `ner.csv` is in the project directory (or upload it to Colab).
4.  **Run the Notebook:**
    Open `BuildingRNNforNER.ipynb` to execute the training and inference steps.

---
*If you find this project helpful, feel free to ⭐ the repo!*
