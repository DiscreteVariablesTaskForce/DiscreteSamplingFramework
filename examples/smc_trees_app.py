import streamlit as st
import json
import graphviz
import os
import matplotlib.pyplot as plt

st.title("🌳 Sequential Monte Carlo Trees Dashboard")

# ----------------------
# Helper Functions
# ----------------------
def load_tree(tree_id):
    with open(f"tree_{tree_id}.json", "r") as file:
        return json.load(file)

def load_feature_names():
    with open("feature_names.json", "r") as f:
        return json.load(f)

def visualize_tree(tree_data, feature_names):
    dot = graphviz.Digraph()
    for node in tree_data["nodes"]:
        if node["is_leaf"]:
            probs = node["probabilities"]
            # Format probabilities as percentages
            prob_str = "\n".join([f"Class {cls}: {prob*100:.1f}%" for cls, prob in probs.items()])
            label = f"Leaf {node['id']}\n{prob_str}"
            dot.node(str(node["id"]), label, shape='box', style='filled', color='lightgreen')
        else:
            feature_idx = node['feature']
            # Use feature name if available; otherwise fallback
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
            label = f"{feature_name} ≤ {node['threshold']:.2f}"
            dot.node(str(node["id"]), label, shape='ellipse', style='filled', color='lightblue')
    for node in tree_data["nodes"]:
        if not node["is_leaf"]:
            dot.edge(str(node["id"]), str(node["left"]), label="True")
            dot.edge(str(node["id"]), str(node["right"]), label="False")
    return dot

def predict_from_tree(tree, input_features):
    """
    Traverse the tree using the input features and return the leaf's probabilities along with
    the list of node IDs visited (the path).
    """
    # Build node lookup dictionary
    nodes = {node["id"]: node for node in tree["nodes"]}
    # Identify the root: we assume it is the first non-leaf node with depth 0
    root = None
    for node in tree["nodes"]:
        if not node.get("is_leaf", False) and node.get("depth", -1) == 0:
            root = node
            break
    if root is None:
        st.error("No root node found in the tree.")
        return {}, []
    
    path = [root["id"]]
    current = root
    while not current.get("is_leaf", False):
        feature_idx = current["feature"]
        threshold = current["threshold"]
        feature_value = input_features[feature_idx]
        if feature_value <= threshold:
            next_id = current["left"]
        else:
            next_id = current["right"]
        path.append(next_id)
        current = nodes[next_id]
    return current["probabilities"], path


def visualize_tree_with_path(tree_data, feature_names, path):
    dot = graphviz.Digraph()

    # First, add all nodes.
    for node in tree_data["nodes"]:
        if node.get("is_leaf", False):
            probs = node.get("probabilities", {})
            prob_str = "\n".join([f"Class {cls}: {prob*100:.1f}%" for cls, prob in probs.items()])
            label = f"Leaf {node['id']}\n{prob_str}"
            # Highlight node if it's on the path.
            if node["id"] in path:
                dot.node(str(node["id"]), label, shape='box', style='filled', color='red')
            else:
                dot.node(str(node["id"]), label, shape='box', style='filled', color='lightgreen')
        else:
            feature_idx = node["feature"]
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
            label = f"{feature_name} ≤ {node['threshold']:.2f}"
            if node["id"] in path:
                dot.node(str(node["id"]), label, shape='ellipse', style='filled', color='red')
            else:
                dot.node(str(node["id"]), label, shape='ellipse', style='filled', color='lightblue')

    # Now add edges. If both nodes in an edge are on the path, highlight the edge.
    for node in tree_data["nodes"]:
        if not node.get("is_leaf", False):
            left_id = node["left"]
            right_id = node["right"]
            # For left edge
            if node["id"] in path and left_id in path:
                dot.edge(str(node["id"]), str(left_id), label="True", color="red", penwidth="2")
            else:
                dot.edge(str(node["id"]), str(left_id), label="True")
            # For right edge
            if node["id"] in path and right_id in path:
                dot.edge(str(node["id"]), str(right_id), label="False", color="red", penwidth="2")
            else:
                dot.edge(str(node["id"]), str(right_id), label="False")
    return dot



# ----------------------
# Build mapping for tree selection
# ----------------------
tree_files = sorted([f for f in os.listdir() if f.startswith("tree_") and f.endswith(".json")])
label_to_tree_id = {}
for filename in tree_files:
    try:
        tree_id = int(filename.split("_")[1].split(".")[0])
    except ValueError:
        continue
    data = load_tree(tree_id)
    stats = data["stats"]
    label = (f"Tree {tree_id} | Nodes: {stats['num_nodes']} | Leaves: {stats['num_leaves']} | "
             f"Depth: {stats['max_depth']} | Accuracy: {stats['accuracy']:.2%}")
    label_to_tree_id[label] = tree_id

# ----------------------
# Create Tabs (without Tree Similarity)
# ----------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌲 Single Tree View",
    "🌲🌳 Compare Trees",
    "📊 Feature Importance",
    "🎯 Interactive Prediction",
    "📈 Overall Performance",
    "🔒 Robustnes Analysis"
])

# ----------------------
# Tab 1: Single Tree View
# ----------------------
with tab1:
    st.header("Single Tree Visualization")
    selected_label = st.selectbox("Select Tree:", list(label_to_tree_id.keys()))
    selected_tree_id = label_to_tree_id[selected_label]
    tree_data = load_tree(selected_tree_id)
    stats = tree_data["stats"]
    st.markdown(f"""
    - **Nodes:** {stats['num_nodes']}  
    - **Leaves:** {stats['num_leaves']}  
    - **Depth:** {stats['max_depth']}  
    - **Accuracy:** {stats['accuracy']:.2%}
    """)
    feature_names = load_feature_names()
    tree_viz = visualize_tree(tree_data, feature_names)
    st.graphviz_chart(tree_viz)

# ----------------------
# Tab 2: Compare Trees (Side-by-Side)
# ----------------------
with tab2:
    st.header("Side-by-Side Tree Comparison")
    col1, col2 = st.columns(2)
    with col1:
        selected_label1 = st.selectbox("Select First Tree:", list(label_to_tree_id.keys()), key='first')
        tree_id1 = label_to_tree_id[selected_label1]
        tree_data1 = load_tree(tree_id1)
        stats1 = tree_data1["stats"]
        st.markdown(f"**Tree {tree_id1} Stats:**  \n- Nodes: {stats1['num_nodes']}  \n- Leaves: {stats1['num_leaves']}  \n- Depth: {stats1['max_depth']}  \n- Accuracy: {stats1['accuracy']:.2%}")
        tree_viz1 = visualize_tree(tree_data1, load_feature_names())
        st.graphviz_chart(tree_viz1)
    with col2:
        selected_label2 = st.selectbox("Select Second Tree:", list(label_to_tree_id.keys()), index=1, key='second')
        tree_id2 = label_to_tree_id[selected_label2]
        tree_data2 = load_tree(tree_id2)
        stats2 = tree_data2["stats"]
        st.markdown(f"**Tree {tree_id2} Stats:**  \n- Nodes: {stats2['num_nodes']}  \n- Leaves: {stats2['num_leaves']}  \n- Depth: {stats2['max_depth']}  \n- Accuracy: {stats2['accuracy']:.2%}")
        tree_viz2 = visualize_tree(tree_data2, load_feature_names())
        st.graphviz_chart(tree_viz2)

# ----------------------
# Tab 3: Feature Importance
# ----------------------
with tab3:
    st.header("Feature Importance across all SMC Trees")
    with open("feature_importance.json", "r") as f:
        importance_dict = json.load(f)
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)
    fig, ax = plt.subplots(figsize=(8, max(4, len(features)*0.4)))
    ax.barh(features[::-1], [imp*100 for imp in importances[::-1]], color="skyblue")
    ax.set_xlabel("Feature Importance (%)")
    ax.set_title("Feature Importance based on Frequency of Usage")
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Importance Values")
    st.dataframe({"Feature": features, "Importance (%)": [round(imp*100,2) for imp in importances]}, use_container_width=True)

# ----------------------
# Tab 4: Interactive Prediction (Exclude "Target")
# ----------------------
with tab4:
    st.header("Interactive Prediction")
    # Load all feature names and remove the outcome "Target"
    all_feature_names = load_feature_names()
    feature_names_for_prediction = [name for name in all_feature_names if name.lower() != "target"]
    st.subheader("Enter Feature Values")
    input_features = []
    # Create a number input for each feature (excluding "Target")
    for i, name in enumerate(feature_names_for_prediction):
        value = st.number_input(f"{name}:", value=0.0, key=f"input_feat_{i}")
        input_features.append(value)
    mode = st.radio("Prediction Mode", options=["Single Tree", "Ensemble"], key="pred_mode")
    if mode == "Single Tree":
        st.subheader("Select a Tree for Prediction")
        single_tree_label = st.selectbox("Tree", list(label_to_tree_id.keys()), key="pred_tree")
        tree_id = label_to_tree_id[single_tree_label]
        tree_data = load_tree(tree_id)
        if st.button("Predict (Single Tree)", key="predict_single"):
            # Get prediction and the path of nodes traversed
            pred, path = predict_from_tree(tree_data, input_features)
            st.write("Predicted probabilities:", pred)
            # Plot the predicted probabilities
            fig, ax = plt.subplots()
            classes = list(pred.keys())
            probs = [pred[cls] for cls in classes]
            ax.bar(classes, probs, color="skyblue")
            ax.set_ylabel("Probability")
            ax.set_title("Single Tree Prediction")
            st.pyplot(fig)
            # Visualize the tree with the prediction path highlighted
            feature_names = load_feature_names()
            tree_viz_path = visualize_tree_with_path(tree_data, feature_names, path)
            st.graphviz_chart(tree_viz_path)
    elif mode == "Ensemble":
        if st.button("Predict (Ensemble)", key="predict_ensemble"):
            all_tree_files = [f for f in os.listdir() if f.startswith("tree_") and f.endswith(".json")]
            ensemble_predictions = []
            for file in all_tree_files:
                try:
                    tree_id = int(file.split("_")[1].split(".")[0])
                except ValueError:
                    continue
                tree_data = load_tree(tree_id)
                pred = predict_from_tree(tree_data, input_features)[0]  # only use prediction, ignore path
                ensemble_predictions.append(pred)
            avg_pred = {}
            for pred in ensemble_predictions:
                for cls, prob in pred.items():
                    avg_pred[cls] = avg_pred.get(cls, 0) + prob
            if ensemble_predictions:
                n = len(ensemble_predictions)
                for cls in avg_pred:
                    avg_pred[cls] /= n
            st.write("Ensemble predicted probabilities:", avg_pred)
            fig, ax = plt.subplots()
            classes = list(avg_pred.keys())
            probs = [avg_pred[cls] for cls in classes]
            ax.bar(classes, probs, color="lightgreen")
            ax.set_ylabel("Probability")
            ax.set_title("Ensemble Prediction")
            st.pyplot(fig)


# ----------------------
# Tab 5: Overall Performance Analysis
# ----------------------
with tab5:
    st.header("Overall Performance Analysis") 

    
    import os, json, numpy as np, matplotlib.pyplot as plt
    
    # Filter only valid tree files that have a "stats" key.
    all_files = [f for f in os.listdir() if f.startswith("tree_") and f.endswith(".json")]
    valid_files = []
    for f in all_files:
        try:
            with open(f, "r") as file:
                data = json.load(file)
            if "stats" in data:
                valid_files.append(f)
            else:
                st.write(f"Skipping {f}: no 'stats' key.")
        except Exception as e:
            st.write(f"Skipping {f} due to error: {e}")
    
    st.write(f"Processing {len(valid_files)} valid tree files.")
    
    # Initialize lists to hold statistics for each tree.
    accuracies = []
    depths = []
    num_nodes = []
    num_leaves = []
    
    for filename in valid_files:
        with open(filename, "r") as file:
            data = json.load(file)
        stats = data.get("stats", {})
        accuracies.append(stats.get("accuracy", np.nan))
        depths.append(stats.get("max_depth", np.nan))
        num_nodes.append(stats.get("num_nodes", np.nan))
        num_leaves.append(stats.get("num_leaves", np.nan))
    
    # Convert lists to numpy arrays for easier handling (optional)
    accuracies = np.array(accuracies)
    depths = np.array(depths)
    num_nodes = np.array(num_nodes)
    num_leaves = np.array(num_leaves)
    
    if len(accuracies) > 0:
        # Create a 2x2 grid of subplots.
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram of Accuracies
        ax[0, 0].hist(accuracies, bins=10, color='skyblue', edgecolor='black')
        ax[0, 0].set_title("Histogram of Accuracies")
        ax[0, 0].set_xlabel("Accuracy")
        ax[0, 0].set_ylabel("Frequency")
        
        # Box Plot of Accuracies
        ax[0, 1].boxplot(accuracies)
        ax[0, 1].set_title("Box Plot of Accuracies")
        ax[0, 1].set_ylabel("Accuracy")
        
        # Scatter Plot: Max Depth vs. Accuracy
        ax[1, 0].scatter(depths, accuracies, color='green')
        ax[1, 0].set_title("Max Depth vs Accuracy")
        ax[1, 0].set_xlabel("Max Depth")
        ax[1, 0].set_ylabel("Accuracy")
        
        # Scatter Plot: Number of Nodes vs. Accuracy
        ax[1, 1].scatter(num_nodes, accuracies, color='purple')
        ax[1, 1].set_title("Number of Nodes vs Accuracy")
        ax[1, 1].set_xlabel("Number of Nodes")
        ax[1, 1].set_ylabel("Accuracy")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additionally, you can display summary statistics in text.
        st.subheader("Summary Statistics")
        st.markdown(f"""
        - **Average Accuracy:** {np.nanmean(accuracies):.2%}
        - **Average Max Depth:** {np.nanmean(depths):.2f}
        - **Average Number of Nodes:** {np.nanmean(num_nodes):.2f}
        - **Average Number of Leaves:** {np.nanmean(num_leaves):.2f}
        """)
    else:
        st.write("No valid tree statistics available.")
        
        
# ----------------------
# Tab 6: Robustness Analysis
# ----------------------
with tab6: 
    st.header("Robustness Analysis")
    
    # Load feature names and remove the outcome "Target"
    all_feature_names = load_feature_names()
    feature_names_for_prediction = [name for name in all_feature_names if name.lower() != "target"]
    
    st.subheader("Enter Base Feature Values")
    base_input_features = []
    for i, name in enumerate(feature_names_for_prediction):
        value = st.number_input(f"{name}:", value=0.0, key=f"robust_input_{i}")
        base_input_features.append(value)
        
    noise_level = st.slider("Noise Level (Standard Deviation)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    num_samples = st.slider("Number of Perturbations", min_value=1, max_value=100, value=20)
    
    mode = st.radio("Prediction Mode", options=["Single Tree", "Ensemble"], key="robust_mode")
    
    predictions_list = []  # List to store predictions from each noisy sample
    
    if mode == "Single Tree":
        selected_label = st.selectbox("Select Tree:", list(label_to_tree_id.keys()), key="robust_tree")
        tree_id = label_to_tree_id[selected_label]
        tree_data = load_tree(tree_id)
        for _ in range(num_samples):
            noise = np.random.normal(0, noise_level, size=len(base_input_features))
            perturbed_input = [base + n for base, n in zip(base_input_features, noise)]
            pred, _ = predict_from_tree(tree_data, perturbed_input)
            predictions_list.append(pred)
    elif mode == "Ensemble":
        all_tree_files = [f for f in os.listdir() if f.startswith("tree_") and f.endswith(".json")]
        for _ in range(num_samples):
            noise = np.random.normal(0, noise_level, size=len(base_input_features))
            perturbed_input = [base + n for base, n in zip(base_input_features, noise)]
            predictions = []
            for file in all_tree_files:
                try:
                    tree_id = int(file.split("_")[1].split(".")[0])
                except ValueError:
                    continue
                tree_data = load_tree(tree_id)
                pred = predict_from_tree(tree_data, perturbed_input)[0]  # only prediction
                predictions.append(pred)
            # Average ensemble predictions across trees
            avg_pred = {}
            for pred in predictions:
                for cls, prob in pred.items():
                    avg_pred[cls] = avg_pred.get(cls, 0) + prob
            if predictions:
                n = len(predictions)
                for cls in avg_pred:
                    avg_pred[cls] /= n
            predictions_list.append(avg_pred)
    
    # Aggregate predictions: convert predictions_list (list of dictionaries) 
    # into a dictionary mapping class -> list of probabilities.
    all_classes = set()
    for pred in predictions_list:
        all_classes.update(pred.keys())
    results = {cls: [] for cls in all_classes}
    for pred in predictions_list:
        for cls in all_classes:
            results[cls].append(pred.get(cls, 0))
    
    # Plot results as box plots for each class
    if results:
        fig, ax = plt.subplots(figsize=(8, 6))
        data_to_plot = [results[cls] for cls in sorted(results.keys())]
        ax.boxplot(data_to_plot, labels=sorted(results.keys()))
        ax.set_xlabel("Class")
        ax.set_ylabel("Predicted Probability")
        ax.set_title("Distribution of Predicted Probabilities under Noise")
        st.pyplot(fig)
        
        st.subheader("Summary Statistics for Predictions")
        for cls in sorted(results.keys()):
            st.write(f"Class {cls}: Mean = {np.mean(results[cls]):.2f}, Std = {np.std(results[cls]):.2f}")
    else:
        st.write("No predictions generated.")

