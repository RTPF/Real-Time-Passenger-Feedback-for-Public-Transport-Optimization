import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import re

# Leer archivo Excel
df = pd.read_excel("Opiniones.xlsx", usecols="S", skiprows=1, names=["comentario"])

# Convertir comentarios a texto
comentarios = df["comentario"].astype(str).tolist()

# Cargar modelo RoBERTa entrenado
modelo_path = "./modelo_roberta_manual"
tokenizer = RobertaTokenizer.from_pretrained(modelo_path)
modelo = RobertaForSequenceClassification.from_pretrained(modelo_path)
modelo.eval()

# Umbral de confianza
umbral_confianza = 0.7

# Mapeo de etiquetas
mapeo_etiquetas = {
    0: "Negativo",
    1: "Positivo",
    2: "Neutro"
}

# Palabras clave por categoría
categorias_keywords = {
    "Chofer": ["chofer", "chófer", "conductor", "manejo", "manejaba", "educado", "grosero", "amable", "maltrato", "choferes", "conductor", "conductores", "exceso de velocidad", "frenan"],
    "Tiempo": ["tarde", "tardado", "retraso", "espera", "puntual", "demora", "tiempo", "se tarda", "tarda mucho en pasar", "tarda en pasar", "paso rapido", "Tardan muchisimo", "es constante su paso", "tardanza", "seguido", "pasan seguido", "hora", "La ruta tarda", "tiempo exagerado", "pasó tarde", "pasó muy tarde", "Se tardo demasiado"],
    "Comodidad": ["cómodo", "incómodo", "apretado", "espacio", "comodidad", "ventanas", "asiento", "lleno", "exceso de pasajeros"],
    "Estado": ["limpio", "sucio", "mal estado", "cuidado", "olor", "funciona bien", "no funciona", "en mal estado", "en buen estado", "en condiciones", "en buenas condiciones", "en malas condiciones"],

}

# Resultados
predicciones = []
confianzas = []
categorias_detectadas = []

# Clasificación + categorización
for texto in comentarios:
    # Clasificación con RoBERTa
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
        probas = torch.softmax(logits, dim=1)
        confianza_max = torch.max(probas).item()
        prediccion = torch.argmax(probas, dim=1).item()

    # Etiqueta final
    if confianza_max < umbral_confianza:
        etiqueta = "Neutro"
    else:
        etiqueta = mapeo_etiquetas.get(prediccion, "desconocido")

    predicciones.append(etiqueta)
    confianzas.append(confianza_max)

    # Categorización temática
    texto_limpio = texto.lower()
    categorias = []

    for categoria, palabras in categorias_keywords.items():
        if any(re.search(rf'\b{re.escape(palabra)}\b', texto_limpio) for palabra in palabras):
            categorias.append(categoria)

    if not categorias and len(texto.strip()) > 2:
        categorias.append("Neutral")
    elif len(texto.strip()) <= 2:
        categorias.append("Sin palabras")

    categorias_detectadas.append(", ".join(categorias))

# Guardar resultados
df["Sentimiento"] = predicciones
df["Confianza"] = confianzas
df["Categoria"] = categorias_detectadas

df.to_csv("Opiniones_yovoy.csv", index=False)

print("✅ Clasificación completada. Archivo guardado.")