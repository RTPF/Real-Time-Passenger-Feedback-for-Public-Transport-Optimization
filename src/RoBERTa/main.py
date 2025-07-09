# Librerías 
import torch
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Verificar si hay una GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Leer los comentarios desde el archivo JSON
with open("./data/set.json", "r", encoding="utf-8") as file:
    comentarios = json.load(file)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Dataset personalizado
class ComentariosDataset(Dataset):
    def __init__(self, data):
        self.encodings = tokenizer([x["text"] for x in data], truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor([x["label"] for x in data])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

dataset = ComentariosDataset(comentarios)
dataloader = DataLoader(dataset, batch_size=6, shuffle=True)  # Tamaño del batch

# Modelo
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
model.to(device)  # Mover el modelo a la GPU
model.train()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Entrenamiento manual
for epoch in range(20):
    print(f"--- Época {epoch+1} ---")
    total_loss = 0
    num_batches = 0
    for batch in dataloader:
        # Mover los datos del batch a la GPU
        batch = {key: val.to(device) for key, val in batch.items()}
        
        optimizer.zero_grad()                    # Reinicia los gradientes anteriores
        outputs = model(**batch)                 # Pasa el batch por el modelo
        loss = outputs.loss   
        total_loss += loss.item()
        num_batches += 1                   # Extrae la pérdida (error)
        print(f"Pérdida: {loss.item():.4f}")     # Imprime el valor de la pérdida
        loss.backward()                          # Calcula gradientes
        optimizer.step()                         # Actualiza los pesos del modelo

    avg_loss = total_loss / num_batches
    print(f"Pérdida promedio en época {epoch+1}: {avg_loss:.4f}")

# Guardar el modelo
model.save_pretrained("./modelo_roberta_manual")
tokenizer.save_pretrained("./modelo_roberta_manual")
print("Modelo entrenado y guardado.")
