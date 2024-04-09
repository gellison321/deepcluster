import torch, numpy as np, matplotlib.pyplot as plt, torch.nn as nn, torch.nn.functional as F, torch.autograd.profiler as profiler, pickle, torch.nn.init as init
from tsshapelet.utils import utils
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import multiprocessing
import inspect


def load_data(name):

    with open(f'/Users/grant/prg/dl/data/timeseries/ucr_uni/{name}/{name}') as f:
        data, labels=[], []

        for line in f:
            line=line.split(',')
            labels.append(int(line[0]))
            data.append(list(map(float, line[1:])))

    X=torch.tensor(data)
    X=(X - X.min()) / (X.max() - X.min())
    Y=torch.tensor(labels, dtype=torch.long)

    with open(f'/Users/grant/prg/dl/data/timeseries/ucr_uni/{name}/{name}_TEST') as f:
        data, labels=[], []

        for line in f:
            line=line.split(',')
            labels.append(int(line[0]))
            data.append(list(map(float, line[1:])))

    X_test=torch.tensor(data)
    X_test=(X_test - X.min()) / (X.max() - X.min())
    Y_test=torch.tensor(labels, dtype=torch.long)

    return X, Y, X_test, Y_test


def load_har(length, test_size=0.2, random_state=42, 
             directory = '/Users/grant/prg/harDEV/HumanActivityData/data/labeled_activity_data'):

    import os
    import pandas as pd

    dataframes = []

    for file in os.listdir(directory):
        if file.endswith('.csv'):
            filepath = os.path.join(directory, file)
            df = pd.read_csv(filepath)
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    X = combined_df['waist_vm'].to_numpy()
    y = combined_df['activity'].to_numpy()

    X = torch.stack([torch.tensor(utils['interpolate'](arr, length)) for arr in X])
    X = (X - X.min()) / (X.max() - X.min())

    enc = {label: i for i, label in enumerate(set(Y))}
    Y = torch.tensor([enc[label] for label in Y])

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    X = (X - X.min()) / (X.max() - X.min())
    X_test = (X_test - X.min()) / (X.max() - X.min())

    return X, X_test, Y, Y_test



# def load_har(length, test_size=0.2, random_state=42):

#     with open('/Users/grant/prg/tsclustering/data/sample_data/X.pickle', 'rb') as f:
#         X = pickle.load(f)

#     X = torch.stack([torch.tensor(utils['interpolate'](arr, length)) for arr in X])
#     X = (X - X.min()) / (X.max() - X.min())

#     with open('/Users/grant/prg/tsclustering/data/sample_data/Y.pickle', 'rb') as f:
#         Y = pickle.load(f)

#     enc = {label: i for i, label in enumerate(set(Y))}
#     Y = torch.tensor([enc[label] for label in Y])

#     X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

#     return X, Y, X_test, Y_test



# Plotting and testing functions



# def plot_model(X, model):

#     colors=['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']*3

#     for i in range(len(X)):
#         x=X[i].reshape(1, -1)

#         with torch.no_grad():
#             y=model.predict(x)

#         plt.plot(x.cpu().detach().numpy().flatten(), color=colors[y.item()])

#     plt.show(block=True)



def plot_data(X, Y):

    colors=['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']*3

    for i in range(len(X)):
        x=X[i].reshape(1, -1)

        with torch.no_grad():
            y=Y[i]

        plt.plot(x.cpu().detach().numpy().flatten(), color=colors[y.item()])

    plt.show(block=True)



def test(model, X, Y):
    
    dataset=torch.utils.data.TensorDataset(X, Y)
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    correct=0
    total=0

    with torch.no_grad():
        for x, y in dataloader:
            x=x.to(model.device, dtype=torch.float32)
            y=y.to(model.device)
            
            y_hat=model.predict(x)  # Predict on the whole batch
            correct += (y_hat == y).sum().item()
            total += y.size(0)

    print(f"Accuracy: {correct / total:.4f}")


def augment_data(X, Y, n=10):
    
    X_augmented, Y_augmented=[], []

    for x, y in zip(X, Y):

        for _ in range(n):
            x_augmented=x + torch.randn_like(x) * 0.01
            X_augmented.append(x_augmented)
            Y_augmented.append(y)

    return torch.stack(X_augmented), torch.tensor(Y_augmented)


def blobs(**kwargs):
    X, y = make_blobs(**{kw : arg for kw, arg in kwargs.items() if kw in inspect.signature(make_blobs).parameters.keys()})
    X, X_test, Y, Y_test = train_test_split(X, y, **{kw : arg for kw, arg in kwargs.items() if kw in inspect.signature(train_test_split).parameters.keys()})
    
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    min_values, _ = torch.min(X, dim=0)
    max_values, _ = torch.max(X, dim=0)

    X = (X - min_values) / (max_values - min_values)
    X_test = (X_test - min_values) / (max_values - min_values)


    return X, X_test, Y, Y_test

def plot_blobs(X, Y, model=None):

    if model==None:

        colors=['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']*3

        for i, (x,y) in enumerate(zip(X[:,0], X[:,1])):
            plt.scatter(x,y, color = colors[Y[i]])
        plt.show(block=True)

        colors=['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']*3

        for i, (x,y) in enumerate(zip(X[:,0], X[:,1])):
            plt.scatter(x,y, color = colors[Y[i]])
        plt.show(block=True)
    
    else:

        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
        for i, (x, y) in enumerate(zip(X, Y)):
            y_hat = model.predict(x)
            plt.scatter(x[0], x[1], color=colors[y_hat])

        for (x,y) in model.centroids.detach().numpy():
            plt.scatter(x, y, color='black', lw = 10)
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = model.predict(torch.tensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        
        plt.show(block=True)