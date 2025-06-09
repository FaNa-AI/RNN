

# 🎵 RNN-Based Spotify Song Popularity Prediction

This repository contains an implementation of a **Recurrent Neural Network (RNN)** using **PyTorch** to predict the popularity of songs on Spotify based on their audio features.

## 📌 Overview

The goal of this project is to predict the `track_popularity` score of a song (on a scale from 0 to 100) using various musical and audio features such as danceability, energy, loudness, etc.

This implementation uses a simple **RNN model** for regression.

## 🔍 Features Used

The model uses the following numerical features from the dataset:

* `danceability`
* `energy`
* `loudness`
* `speechiness`
* `acousticness`
* `instrumentalness`
* `liveness`
* `valence`
* `tempo`

**Target variable:**

* `track_popularity`

## 📁 Dataset

The dataset file should be named `spotify_songs.csv`.
Make sure to update the path in the script if needed:

```python
df = pd.read_csv(r'c:\faezeh\MachineLearning\ForthPractice\rnn\rnn/spotify_songs.csv')
```

Missing data is dropped before training.

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* pandas
* numpy
* scikit-learn
* matplotlib

Install the dependencies with:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## 🏗 Model Architecture

* **Input:** (Batch, Sequence Length = 1, Features)
* **RNN Layer:**

  * `input_size = 9` (number of features)
  * `hidden_size = 64`
  * `num_layers = 1`
* **Fully Connected Output Layer:** Maps the RNN output to a single regression value (song popularity).

## 🧠 Training

* Optimizer: `Adam`
* Loss Function: `Mean Squared Error (MSE)`
* Epochs: 15
* Batch Size: 64

Training loss is printed per epoch and also plotted at the end.

## 📈 Evaluation

After training, the model is evaluated on the test set and the final **MSE loss** is printed.

The training loss over epochs is visualized using a simple line plot.

## 📊 Output Example

![Training Loss Plot](example_loss_plot.png)

> (Use `plt.savefig("example_loss_plot.png")` in the script if you'd like to save the plot.)

## 📂 File Structure

```
├── spotify_songs.csv       # Dataset
├── rnn_regression.py       # Main script (your code)
└── README.md               # Project documentation
```

## ▶️ How to Run

```bash
python rnn_regression.py
```

