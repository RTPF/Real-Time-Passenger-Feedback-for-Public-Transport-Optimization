import os
import csv
import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Crear carpeta 'data' si no existe
os.makedirs("data", exist_ok=True)

# Cargar modelo entrenado
model_path = "./modelo_roberta_manual"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Umbral de confianza
threshold = 0.6

# Cargar comentarios desde el archivo JSON
with open("./data/comentarios.json", "r", encoding="utf-8") as file:
    nuevos_comentarios = json.load(file)

# Clasificar los comentarios
resultados = []
errores = 0

# Etiquetas opcionales
etiquetas_map = {
    0: "Negativo",
    1: "Positivo",
    2: "Mixto"
}

for comentario in nuevos_comentarios:
    texto = comentario["text"]
    etiqueta_real = comentario["label"]
    
    # Tokenizar y pasar por el modelo
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probas = torch.softmax(logits, dim=-1)
        max_prob, prediccion_tensor = torch.max(probas, dim=-1)
        max_prob = max_prob.item()
        prediccion = prediccion_tensor.item()

    if max_prob < threshold:
        clasificacion = "Requiere revisión humana"
        correcto = False  # No podemos saber si fue correcta
    else:
        clasificacion = etiquetas_map.get(prediccion, f"Clase {prediccion}")
        correcto = prediccion == etiqueta_real

        if not correcto:
            errores += 1

    resultados.append({
        "comentario": texto,
        "clasificacion": clasificacion,
        "etiqueta_real": etiquetas_map.get(etiqueta_real, f"Clase {etiqueta_real}"),
        "correcto": correcto
    })

# Guardar en CSV dentro de la carpeta "data"
csv_path = os.path.join("data", "clasificados.csv")
with open(csv_path, mode="w", encoding="utf-8", newline="") as archivo:
    writer = csv.DictWriter(archivo, fieldnames=["comentario", "clasificacion", "etiqueta_real", "correcto"])
    writer.writeheader()
    for fila in resultados:
        writer.writerow(fila)

# Imprimir el número de errores
print(f"Clasificación completada. Resultados guardados en: {csv_path}")
print(f"Número de errores: {errores}")