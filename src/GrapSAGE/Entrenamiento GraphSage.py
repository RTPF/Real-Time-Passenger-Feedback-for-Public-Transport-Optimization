import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
folder_name = "Grafos_Rutas_Optimizado"
folder_path = os.path.join(desktop, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

df_feedback = pd.read_csv(r"C:\Users\AlexisVM\Downloads\pasajeros_feedback.csv", encoding="latin-1")
df_feedback.dropna(inplace=True)

df_rutas = pd.read_csv(r"C:\Users\AlexisVM\Downloads\adjusted_frequencies_optimized.csv", encoding="latin-1")

df = pd.merge(df_feedback, df_rutas, on="route_id", how="inner")

df["route_id"] = df["route_id"].astype(str)
route_to_index = {route: idx for idx, route in enumerate(sorted(df["route_id"].unique()))}
df["route_id_num"] = df["route_id"].map(route_to_index)

df["passenger_id_num"] = df["passenger_id"].astype("category").cat.codes

edge_index = []
node_features = {}

for _, row in df.iterrows():
    route = row["route_id_num"]
    passenger = row["passenger_id_num"]
    
    # Agregar conexión entre pasajero y ruta
    edge_index.append([route, passenger])
    
    node_features[route] = [row["adjusted_frequency"], row["simulated_satisfaction"], row["simulated_wait_time"]]
    node_features[passenger] = [row["initial_satisfaction"], row["initial_wait_time"], row["proportion_positive"]]

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
x = torch.tensor([node_features[n] for n in sorted(node_features)], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(x.size(1), 64)
        self.conv2 = SAGEConv(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_ids, test_ids = train_test_split(list(route_to_index.values()), test_size=0.2, random_state=42)

train_ids_tensor = torch.tensor(train_ids, dtype=torch.long)
test_ids_tensor = torch.tensor(test_ids, dtype=torch.long)

for epoch in range(10):
    optimizer.zero_grad()
    pred = model(data)
    loss = F.mse_loss(pred[train_ids_tensor], x[train_ids_tensor][:, 1])  # Usar columna de satisfacción simulada
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

predicted_satisfaction = model(data).detach().numpy().flatten()
target_df = pd.DataFrame({
    "route_id": list(route_to_index.keys()), 
    "predicted_satisfaction": predicted_satisfaction[:len(route_to_index)]
})
target_df["priority_flag"] = target_df["predicted_satisfaction"] < 0.5

csv_path = os.path.join(folder_path, "rutas_prioritarias_optimizado.csv")
target_df.to_csv(csv_path, index=False)

print(f"Proceso completado. Se generó '{csv_path}' con las rutas prioritarias.")