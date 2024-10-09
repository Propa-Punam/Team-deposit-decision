# Term Deposit Decision

This project demonstrates a simple feedforward neural network to predict whether a customer will subscribe to a term deposit. The model uses the Bank Marketing dataset, and the neural network is implemented from scratch without external deep learning libraries (except for basic dependencies such as numpy and pandas).

## Dataset

The dataset used is the Bank Marketing Dataset. It is available on UCI Machine Learning Repository. The dataset includes customer data and whether they subscribed to a term deposit (y = 1 if yes, 0 if no).

The dataset (bank-full.csv) should be placed in the `/data` folder.

### Dataset Features
- `age`, `day`, `campaign`: features used in the model for prediction.
- `y`: target label indicating whether a customer subscribed to the term deposit.

## Project Structure
```
/project-root
│
├── /data                    # Directory containing datasets
│   └── bank-full.csv         # Dataset file
│
├── /src                     # Source code directory
│   └── model.py              # Contains neural network implementation
│   └── utils.py              # Helper functions (optional)
│
├── main.py                  # Main script to train and evaluate the model
│
├── requirements.txt         # List of dependencies
│
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add the dataset:
   - Download the Bank Marketing dataset and place it in the `data/` folder.

## Usage

1. **Prepare and train the model:** Run the `main.py` script to train the neural network and evaluate its accuracy.
```bash
python main.py
```

2. **Prediction:** The `main.py` script will train the model and make predictions on the same data. You can modify the code to make predictions on new data as needed.

## Model Description

The neural network consists of the following:
- **Input layer**: Takes the input features (age, day, and campaign).
- **Hidden layer**: 10 nodes with sigmoid activation function.
- **Output layer**: 1 node, predicting whether the customer will subscribe (1) or not (0).

The model uses forward propagation, backward propagation, and updates the weights through gradient descent.

## Results

After training the model, it outputs the accuracy of the prediction. For example:
```
Accuracy: 88%
```

## License

This project is licensed under the Apache License 2.0 License. See the LICENSE file for details.
