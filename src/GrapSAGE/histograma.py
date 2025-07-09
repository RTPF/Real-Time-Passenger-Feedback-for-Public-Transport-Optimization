import pandas as pd
import matplotlib.pyplot as plt
import os

ruta_csv = r"C:\Users\AlexisVM\Downloads\pasajeros_feedback.csv"
df = pd.read_csv(ruta_csv, encoding="latin-1")

satisfaction_map = {
    "1.- Muy insatisfecho": "Very Dissatisfied",
    "2.- Insatisfecho": "Dissatisfied",
    "3.- Regular": "Neutral",
    "4.- Satisfecho": "Satisfied",
    "5.- Muy satisfecho": "Very Satisfied"
}
df["satisfaction_en"] = df["satisfaction"].map(satisfaction_map)

counts = df["satisfaction_en"].value_counts().reindex([
    "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"
])

plt.figure(figsize=(8, 5))
plt.bar(counts.index, counts.values, color="skyblue")
plt.xlabel("Satisfaction Level")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=30)
plt.tight_layout()

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "satisfaction_histogram.svg")
plt.savefig(desktop_path, format="svg")
plt.close()

print(f"Gráfico guardado en: {desktop_path}")