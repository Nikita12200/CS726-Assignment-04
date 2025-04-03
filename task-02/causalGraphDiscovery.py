import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from itertools import combinations

# Custom Dataset for Observational Data
class ObservationalData(data.Dataset):
    def __init__(self, data):
        if isinstance(data, torch.Tensor):
            self.data = data.clone().detach().float()
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Neural Network Model (Multilayer Perceptron)
class MultivarMLP(nn.Module):
    def __init__(self, num_vars, hidden_dims):
        # TODO: Students should implement the neural network architecture
        # Hint: Create a sequential model with linear layers, ReLU activations, and an output layer for mean and log-variance
        pass
    
    def forward(self, x, sample_matrix):
        # TODO: Students should implement the forward pass of the neural network
        # Hint: Apply the sample matrix to mask the input, pass it through the layers, and return mean and log-variance
        pass

def create_model(num_vars, hidden_dims):
    # TODO: Students should implement the creation of the neural network model
    # Hint: Instantiate and return a MultivarMLP model with the given number of variables and hidden dimensions
    pass

class ENCO(object):
    def __init__(self, 
                 data_obs,  # Observational data as a numpy array
                 hidden_dims=[64, 32],
                 lr_model=5e-3,
                 betas_model=(0.9, 0.999),
                 weight_decay=0.0,
                 lr_gamma=2e-2,
                 betas_gamma=(0.9, 0.9),
                 lr_theta=1e-1,
                 betas_theta=(0.9, 0.999),
                 model_iters=1000,
                 graph_iters=100,
                 batch_size=128,
                 lambda_sparse=0.001,
                 sample_size_obs=5000):
        """
        Creates ENCO object for performing causal structure learning with observational data only.
        Students will need to implement the initialization logic.
        """
        # Normalize the observational data to have zero mean and unit variance
        scaler = StandardScaler()
        data_obs = scaler.fit_transform(data_obs)
        self.data_obs = torch.tensor(data_obs, dtype=torch.float32)
        self.num_vars = self.data_obs.shape[1]
        self.variable_names = [f'X{i+1}' for i in range(self.num_vars)]

        # Truncate the dataset if it exceeds the specified sample size
        if len(self.data_obs) > sample_size_obs:
            self.data_obs = self.data_obs[:sample_size_obs]

        # TODO: Students should implement the initialization of the dataset, model, and optimizers
        # Hint: Create an ObservationalData dataset, a DataLoader, the MultivarMLP model, and optimizers for the model, gamma, and theta
        self.obs_data_loader = None  # Placeholder for DataLoader
        self.model = None  # Placeholder for the neural network model
        self.model_optimizer = None  # Placeholder for the model optimizer
        self.gamma = None  # Placeholder for gamma parameters
        self.gamma_optimizer = None  # Placeholder for gamma optimizer
        self.theta = None  # Placeholder for theta parameters
        self.theta_optimizer = None  # Placeholder for theta optimizer

        # Save hyperparameters for use in training
        self.model_iters = model_iters
        self.graph_iters = graph_iters
        self.lambda_sparse = lambda_sparse
        self.iter_time = -1
        self.dist_fit_time = -1
        self.epoch = 0

        # Debugging info (students can use this to verify their dataset and model)
        print(f'Dataset size:\n- Observational: {len(self.data_obs)}')

    def discover_graph(self, num_epochs=200):
        """
        Main training function for ENCO. Students should implement the training loop.
        """
        # TODO: Students should implement the training loop for ENCO
        # Hint: Iterate over epochs, calling distribution_fitting_step and graph_fitting_step, and print progress
        pass
        return self.get_binary_adjmatrix()

    def distribution_fitting_step(self):
        """
        Performs one iteration of distribution fitting. Students should implement this method.
        """
        # TODO: Students should implement the distribution fitting step
        # Hint: Train the neural network to fit the conditional distributions using the sample matrix derived from gamma and theta
        pass

    def graph_fitting_step(self):
        """
        Performs one iteration of graph fitting. Students should implement this method.
        """
        # TODO: Students should implement the graph fitting step
        # Hint: Update gamma and theta parameters to infer the causal graph, using the neural network to compute reconstruction loss
        pass

    def get_binary_adjmatrix(self):
        """
        Returns the predicted binary adjacency matrix. Students should implement this method.
        """
        # TODO: Students should implement the computation of the binary adjacency matrix
        # Hint: Convert gamma and theta to binary values (e.g., using a threshold) and combine them to form the adjacency matrix
        return np.zeros((self.num_vars, self.num_vars))  # Placeholder return

    def visualize_graph(self, adj_matrix, method="ENCO"):
        """
        Visualizes the adjacency matrix as a heatmap and saves it to a file.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(adj_matrix, cmap='hot' if method == "ENCO" else 'coolwarm', interpolation='nearest')
        plt.colorbar(label='Edge Probability' if method == "ENCO" else 'Edge Confidence')
        plt.xticks(np.arange(self.num_vars), self.variable_names, rotation=45)
        plt.yticks(np.arange(self.num_vars), self.variable_names)
        plt.title(f'Inferred Causal Graph ({method}, Epoch {self.epoch if method == "ENCO" else ""})')
        plt.savefig(f'inferred_graph_{method.lower()}.png' if method != "True" else "inferred_graph_true.png")
        plt.close()

# PC Algorithm Functions (to be implemented by students)
def partial_correlation(data, x, y, z):
    """
    Computes the partial correlation between variables x and y given a set of conditioning variables z.
    Students should implement this method.
    """
    # TODO: Students should implement the partial correlation calculation
    # Hint: Compute the correlation between x and y while controlling for the variables in z
    return 0  # Placeholder return

def independence_test(data, x, y, z):
    """
    Performs an independence test between variables x and y given a set of conditioning variables z.
    Students should implement this method.
    """
    # TODO: Students should implement the independence test
    # Hint: Use partial correlation to compute a p-value and return a confidence score (1 - p_value)
    return 1.0, 0  # Placeholder return (confidence, partial correlation)

def pc_algorithm(data, alpha=0.05, max_condition_set_size=3):
    """
    Implements the PC algorithm to infer a causal graph from observational data.
    Students should implement this method.
    """
    # TODO: Students should implement the PC algorithm
    # Hint: Start with a fully connected graph, remove edges based on conditional independence tests, and orient edges using the PC algorithm rules
    n_vars = data.shape[1]
    return   # Placeholder return


def main():
    # Load Observational Data
    csv_file = "X1.csv"
    try:
        data = pd.read_csv(csv_file, delimiter=',')
        data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        data_obs = data.values
        print(f"Loaded data from {csv_file} with shape {data_obs.shape}")
        
        if data.isnull().sum().sum() > 0:
            print("WARNING: Missing values detected! Imputing with mean.")
            data_obs = np.nan_to_num(data_obs, nan=np.nanmean(data_obs))
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
            print("ERROR: Not all columns are numeric.")
            sys.exit(1)
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        sys.exit(1)

    # Load True Adjacency Matrix
    true_csv_file = "W1_true.csv"
    try:
        true_data = pd.read_csv(true_csv_file, delimiter=',', header=None)
        true_adj_matrix = true_data.values
        print(f"Loaded true adjacency matrix from {true_csv_file} with shape {true_adj_matrix.shape}")
        
        if true_adj_matrix.shape != (15, 15):
            print("ERROR: True adjacency matrix must be 15x15 to match the number of variables.")
            sys.exit(1)
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {true_csv_file}")
        sys.exit(1)

    # Generate variable names for labeling the axes in visualizations
    variable_names = [f'X{i+1}' for i in range(15)]

    # Visualize and save the true adjacency matrix for reference
    enco_instance = ENCO(data_obs=data_obs, batch_size=64, lambda_sparse=0.001)  # Create instance for visualization
    enco_instance.visualize_graph(true_adj_matrix, method="True")
    pd.DataFrame(true_adj_matrix, index=variable_names, columns=variable_names).to_csv("true_adj_matrix.csv")
    print(f"True adjacency matrix saved to true_adj_matrix.csv")

    # ENCO Implementation
    enco_module = ENCO(data_obs=data_obs, batch_size=64, lambda_sparse=0.001)
    enco_adj_matrix = enco_module.discover_graph(num_epochs=100)
    enco_module.visualize_graph(enco_adj_matrix, method="ENCO")
    pd.DataFrame(enco_adj_matrix, index=variable_names, columns=variable_names).to_csv("enco_pred_causal_graph.csv")
    print(f"ENCO predicted adjacency matrix (probabilities) saved to enco_pred_causal_graph.csv")

    # PC Algorithm Implementation
    pc_adj_matrix = pc_algorithm(data_obs, alpha=0.05)
    enco_instance.visualize_graph(pc_adj_matrix, method="PC")  # Reuse ENCO instance for visualization
    pd.DataFrame(pc_adj_matrix, index=variable_names, columns=variable_names).to_csv("pc_pred_causal_graph.csv")
    print(f"PC predicted adjacency matrix (confidence scores) saved to pc_pred_causal_graph.csv")

    # Output adjacency matrices for comparison
    print("True Adjacency Matrix:\n", true_adj_matrix)
    print("ENCO Adjacency Matrix (Probabilities):\n", enco_adj_matrix)
    print("PC Adjacency Matrix (Confidence Scores):\n", pc_adj_matrix)

if __name__ == '__main__':
    main()