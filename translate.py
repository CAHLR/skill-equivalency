import torch
import copy
from zlib import crc32
import numpy as np
import pandas as pd


def string_to_float(string):
    '''
    Hash function that uniformly maps any string to a random number in (0,1)
    Used to split training and test sets
    '''
    return float(crc32(string.encode("utf-8")) & 0xffffffff) / 2**32


def split_labels(labels_path, test_ratio=0.2):
    '''
    Split labels into training and test sets
    New splits are saved under the same folder as labels_path
    '''
    labels = pd.read_csv(labels_path)
    labels["random"] = labels["source"].apply(string_to_float)
    training_labels = labels[labels["random"] > test_ratio].drop(columns="random")
    test_labels = labels[labels["random"] <= test_ratio].drop(columns="random")
    
    training_labels_path = labels_path[:-4] + "_training.csv"
    test_labels_path = labels_path[:-4] + "_test.csv"
    
    training_labels.to_csv(training_labels_path, index=False)
    test_labels.to_csv(test_labels_path, index=False)
    return training_labels_path, test_labels_path


class translation_model(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True).to(device)

    def l2_normalization(self, x):
        return torch.div(x, torch.sqrt(torch.mul(x, x).sum(axis=1).unsqueeze(1)))

    def translate(self, src):
        return self.l2_normalization(self.linear(src))

    def get_cos_similarity(self, x, y):
        x = self.l2_normalization(x)
        y = self.l2_normalization(y)
        new_x = x.unsqueeze(1).repeat(1, y.size(0), 1)
        new_y = y.unsqueeze(0).repeat(x.size(0), 1, 1)
        return torch.mul(new_x, new_y).sum(axis=2)


def learn_translation_model(src_vectors, dst_vectors, labels, device,
                            num_epochs, val_ratio, max_val_non_decreasing_epochs, verbose):
    labels["src_index"] = labels["source"].apply(lambda x: src_vectors.index.tolist().index(x))
    labels["dst_index"] = labels["destination"].apply(lambda x: dst_vectors.index.tolist().index(x))
    val_index = np.random.permutation(labels.shape[0])[:int(labels.shape[0] * val_ratio)]
    training_labels = labels.loc[~labels.index.isin(val_index), ["src_index", "dst_index"]]
    val_labels = labels.loc[labels.index.isin(val_index), ["src_index", "dst_index"]]
    
    model = translation_model(in_dim=src_vectors.shape[1], out_dim=dst_vectors.shape[1], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    src = torch.tensor(src_vectors.to_numpy(), dtype=torch.float).to(device)
    dst = torch.tensor(dst_vectors.to_numpy(), dtype=torch.float).to(device)
    
    # train the model
    best_val_loss = float("inf")
    best_model = model
    val_non_decreasing_epochs = 0
    for i in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        src_translated = model.translate(src)
        cos_sim = model.get_cos_similarity(src_translated, dst)
   
        loss = torch.sum(1 - cos_sim[list(training_labels['src_index']), list(training_labels['dst_index'])])
        loss.backward()
        optimizer.step()
        
        # check if val loss is decreasing
        val_loss = torch.sum(1 - cos_sim[list(val_labels['src_index']), list(val_labels['dst_index'])])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            val_non_decreasing_epochs = 0
        else:
            val_non_decreasing_epochs += 1
        
        if verbose:
            print(f"Epochs {i}, training loss {loss:.4f}, validation loss {val_loss:.4f}, val loss not decreased in last {val_non_decreasing_epochs} epochs")
        
        if val_non_decreasing_epochs >= max_val_non_decreasing_epochs and len(val_labels) > 0:
            break
    
    return best_model


def generate_translated_source_vectors(
        src_vectors_path, dst_vectors_path, training_labels_path, device,
        num_epochs=1000, val_ratio=0.2, max_val_non_decreasing_epochs=100, verbose=False):
    src_vectors = pd.read_csv(src_vectors_path, index_col=0)
    dst_vectors = pd.read_csv(dst_vectors_path, index_col=0)
    training_labels = pd.read_csv(training_labels_path)
    
    model = learn_translation_model(
        src_vectors, dst_vectors, training_labels, device,
        num_epochs, val_ratio, max_val_non_decreasing_epochs, verbose)
    model.eval()
    
    src_tensor = torch.tensor(src_vectors.to_numpy(), dtype=torch.float).to(device)
    src_translated = model.translate(src_tensor).detach().cpu().numpy()
    
    src_translated = pd.DataFrame(src_translated, index=src_vectors.index)
    src_translated.index.name = "skill"
    dst_vectors_name = dst_vectors_path.split("/")[-1][:-4]
    src_translated_path = src_vectors_path[:-4] + f"_translated_to_{dst_vectors_name}.csv"
    
    src_translated.to_csv(src_translated_path)
    return src_translated_path
