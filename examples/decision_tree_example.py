# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""
import copy
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_wine
import numpy as np
#from discretesampling.domain.decision_tree.helper_functions import *
#from art.attacks import DecisionTreeAttack
#from art.classifiers import SklearnClassifier




df = pd.read_csv(r"C:\Users\avarsi88\PycharmProjects\Parallel_SMC_sampler_Discrete_Variables\examples\datasets_smc_mcmc_CART\LiverDisorder.csv")

df = df.dropna()
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()


acc = []
for i in range(1):
    acc_per_tree = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
    a = 10# size of the tree
    #b = 5
    target = dt.TreeTarget(a)
    initialProposal = dt.TreeInitialProposal(X_train, y_train)
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
    try:
        treeSMCSamples, current_possibilities_for_predictions, logweights = dtSMC.sample(10, 5, a, resampling= "systematic")#residual,#systematic, knapsack, min_error, variational, min_error_imp, CIR
        #(100number of iterations,25number of trees)
        smcLabels, prob, leaf_prob, leaf_for_prediction = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
        
        smcLabel_per_tree = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=False)
        smcAccuracy = [dt.accuracy(y_test, smcLabels)]
        print("SMC mean accuracy: ", np.mean(smcAccuracy))
        acc.append(smcAccuracy)
        for label in smcLabel_per_tree:
            acc_per_tree.append(dt.accuracy(y_test, label))
        
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
    
print("overall acc for 10 mc runs is: ", np.mean(acc))






feature_names = list(df.columns)


import json


with open("feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)
    
    
import numpy as np


import json

def save_tree_to_json(tree, leaf_probs, accuracy, tree_id):
    nodes_json = []

    leaf_set = set(int(leaf) for leaf in tree.leafs)
    leaf_id_to_index = {int(leaf): idx for idx, leaf in enumerate(tree.leafs)}

    # Store all internal nodes clearly
    for node in tree.tree:
        node_id, left_id, right_id, feature, threshold, depth = node
        node_id = int(node_id)

        node_json = {
            "id": node_id,
            "left": int(left_id),
            "right": int(right_id),
            "feature": int(feature),
            "threshold": float(threshold),
            "depth": int(depth),
            "is_leaf": False
        }
        nodes_json.append(node_json)

    # Store leaf nodes separately
    for leaf_id in tree.leafs:
        leaf_idx = tree.leafs.index(leaf_id)
        probs = leaf_probs[leaf_id_to_index[int(leaf_id)]]
        probs_dict = {str(cls): float(prob) for cls, prob in probs.items()}

        leaf_json = {
            "id": int(leaf_id),
            "is_leaf": True,
            "probabilities": probs_dict,
            # Set depth if available (or remove if not)
        }

        nodes_json.append(leaf_json)

    # Compute depth clearly from all nodes and leafs
    max_depth = max(node[5] for node in tree.tree)

    tree_data = {
        "nodes": nodes_json,
        "leafs": [int(leaf) for leaf in tree.leafs],
        "stats": {
            "num_nodes": len(tree.tree),
            "num_leaves": len(tree.leafs),
            "max_depth": int(max_depth),
            "accuracy": float(accuracy)
        }
    }

    with open(f"tree_{tree_id}.json", "w") as f:
        json.dump(tree_data, f, indent=4)

# Save for all trees clearly:
for idx, (tree_sample, leaf_probs, acc) in enumerate(zip(treeSMCSamples, current_possibilities_for_predictions, acc_per_tree)):
    save_tree_to_json(tree_sample, leaf_probs, acc, idx)
    
    
import os
import json
import numpy as np

# Load feature names 
feature_names = json.load(open("feature_names.json"))

# Filter valid tree files
tree_files = []
for f in os.listdir():
    if f.startswith("tree_") and f.endswith(".json"):
        try:
            with open(f, "r") as file:
                data = json.load(file)
            # Check if 'nodes' exists and is a list
            if "nodes" in data and isinstance(data["nodes"], list):
                tree_files.append(f)
            else:
                print(f"Skipping {f} because it lacks a proper 'nodes' key.")
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")


# Initialize feature counts for all features
feature_counts = np.zeros(len(feature_names))

# Loop through the valid tree JSON files to compute feature counts
for filename in tree_files:
    with open(filename, "r") as file:
        data = json.load(file)
        for node in data["nodes"]:
            # Process only internal nodes
            if not node.get("is_leaf", False):
                feature_idx = node["feature"]
                feature_counts[feature_idx] += 1

# Compute importance as a percentage
feature_importance = feature_counts / feature_counts.sum()

# Save importance to a JSON file for later use
importance_dict = {feature_names[i]: float(importance) for i, importance in enumerate(feature_importance)}

with open("feature_importance.json", "w") as f:
    json.dump(importance_dict, f, indent=4)


import json
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# List all files that match the basic pattern
all_tree_files = [f for f in os.listdir() if f.startswith("tree_") and f.endswith(".json")]

# Filter only those that contain a "nodes" key
valid_tree_files = []
for filename in all_tree_files:
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        if "nodes" in data and isinstance(data["nodes"], list):
            valid_tree_files.append(filename)
        else:
            print(f"Skipping {filename} because it lacks a valid 'nodes' key.")
    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")


# Now use only the valid files for your feature usage matrix:
feature_usage_matrix = []

for filename in valid_tree_files:
    with open(filename, "r") as file:
        data = json.load(file)
        feature_counts = {}
        # Count occurrences of each feature from internal nodes
        for node in data["nodes"]:
            if not node.get("is_leaf", False):
                feature = node["feature"]
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        feature_usage_matrix.append(feature_counts)
