import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Cargar el modelo y el tokenizer desde la carpeta guardada
model_path = "./modelo_roberta_manual"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Función para predecir
def predecir(texto):
    # Tokenizar el texto
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Pasar el texto por el modelo
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener las probabilidades
    logits = outputs.logits
    probabilidades = torch.softmax(logits, dim=1)
    
    # Obtener la clase predicha
    clase_predicha = torch.argmax(probabilidades, dim=1).item()
    return clase_predicha, probabilidades

# Interactuar desde la consola
if __name__ == "__main__":
    print("Escribe un mensaje para clasificar (escribe 'salir' para terminar):")
    while True:
        texto = input(">> ")
        if texto.lower() == "salir":
            break
        clase, probs = predecir(texto)
        print(f"Clase predicha: {clase} (Probabilidades: {probs.numpy()})")