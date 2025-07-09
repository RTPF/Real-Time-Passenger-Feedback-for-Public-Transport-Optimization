import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

path_feedback = r"C:\Users\AlexisVM\Downloads\pasajeros_feedback.csv"
path_rutas = r"C:\Users\AlexisVM\Downloads\adjusted_frequencies_optimized.csv"

df_feedback = pd.read_csv(path_feedback, encoding="latin-1").dropna()
df_rutas = pd.read_csv(path_rutas, encoding="latin-1")
df = pd.merge(df_feedback, df_rutas, on="route_id", how="inner")
df["adjusted_frequency"] = df["adjusted_frequency"].fillna(df["adjusted_frequency"].median())

route_features = ["adjusted_frequency"]
passenger_features = ["proportion_positive"]
scaler = StandardScaler()
df[route_features] = scaler.fit_transform(df[route_features])
df[passenger_features] = scaler.fit_transform(df[passenger_features])

df["route_id"] = df["route_id"].astype(str)
route_map = {rid: i for i, rid in enumerate(sorted(df["route_id"].unique()))}
df["route_idx"] = df["route_id"].map(route_map)
df["passenger_idx"] = df["passenger_id"].astype("category").cat.codes + len(route_map)

edge_index = []
node_features = {}
labels = {}

for _, row in df.iterrows():
    r_idx = row["route_idx"]
    p_idx = row["passenger_idx"]
    edge_index.extend([[r_idx, p_idx], [p_idx, r_idx]])

    if r_idx not in node_features:
        node_features[r_idx] = row[route_features].tolist()
        satisfaction_map = {"Muy satis": 1.0, "Satisfech": 0.8, "Regular": 0.5, "Insatisfech": 0.3, "Muy insal": 0.1}
        labels[r_idx] = 1 if satisfaction_map.get(row["satisfaction"], 0.5) >= 0.5 else 0
    if p_idx not in node_features:
        node_features[p_idx] = row[passenger_features].tolist()

max_dim = max(len(v) for v in node_features.values())
for k in node_features:
    while len(node_features[k]) < max_dim:
        node_features[k].append(0.0)

x = torch.tensor([node_features[i] for i in range(len(node_features))], dtype=torch.float)
x += torch.normal(mean=0.0, std=0.025, size=x.shape)  # Ruido más fuerte

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edge_index)

total_nodes = len(node_features)
is_route_node = torch.zeros(total_nodes, dtype=torch.bool)
target_tensor = torch.zeros(total_nodes)

for idx in route_map.values():
    is_route_node[idx] = True
    target_tensor[idx] = labels[idx]

print("\n🔎 Verificación de distribución de etiquetas:")
print("Clase 0:", (target_tensor == 0).sum().item())
print("Clase 1:", (target_tensor == 1).sum().item())

route_indices = list(route_map.values())
route_labels = [labels[i] for i in route_indices]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(route_indices, route_labels)):
    train_route_indices = torch.tensor([route_indices[i] for i in train_idx])
    test_route_indices = torch.tensor([route_indices[i] for i in test_idx])

    print(f"\n📌 Fold {fold+1}: Validación de separación de datos")
    print("Test set - Clase 0:", (target_tensor[test_route_indices] == 0).sum().item())
    print("Test set - Clase 1:", (target_tensor[test_route_indices] == 1).sum().item())

    class GATBinaryClassifier(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.conv1 = GATConv(input_dim, 32, heads=1, dropout=0.75)
            self.conv2 = GATConv(32, 16, heads=1, dropout=0.75)
            self.fc_out = torch.nn.Linear(16, 1)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.fc_out(x)
            return torch.sigmoid(x.squeeze())

    model = GATBinaryClassifier(input_dim=x.size(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.8]))

    model.train()
    for epoch in range(140):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output[train_route_indices], target_tensor[train_route_indices].float())
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        pred_labels = (model(data).detach().numpy()[is_route_node.numpy()] >= 0.5).astype(int)
        print("\n📌 Evaluación detallada del modelo:")
        print(classification_report(target_tensor[is_route_node].numpy(), pred_labels, zero_division=1))

    accuracies.append(accuracy_score(target_tensor[is_route_node].numpy(), pred_labels))

print(f"\n✅ Exactitud promedio: {sum(accuracies) / len(accuracies) * 100:.2f}%")