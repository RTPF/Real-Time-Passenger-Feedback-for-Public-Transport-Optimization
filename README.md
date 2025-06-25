# Real-Time Passenger Feedback for Public Transport Optimization

Este proyecto busca optimizar el transporte público mediante el análisis en tiempo real de los comentarios y opiniones de los pasajeros. Integra procesamiento de lenguaje natural, aprendizaje por refuerzo y análisis de grafos para mejorar la experiencia del usuario y la eficiencia del sistema, utilizando el feedback de los usuarios como base para la toma de decisiones.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Dataset](#dataset)
- [RoBERTa](#roberta)
- [Q-learning](#q-learning)
- [GraphSAGE](#graphsage)

## Descripción General

El sistema integra modelos de clasificación de sentimientos, aprendizaje por refuerzo y análisis de grafos para procesar y aprovechar el feedback de los usuarios del transporte público. El flujo general es:

1. Recolección de datos y comentarios de usuarios.
2. Clasificación de los comentarios usando [RoBERTa](src/RoBERTa/) (positivo, negativo, neutral).
3. Optimización de recompensas y políticas mediante [Q-learning](src/Q-learning/).
4. Análisis de relaciones y patrones entre usuarios con [GraphSAGE](src/GraphSAGE/).


## Dataset

En la carpeta [`data`](data) se encuentra el dataset utilizado, creado a partir de datos proporcionados por la **Coordinación General de Movilidad** ([CMOV](https://www.aguascalientes.gob.mx/cmov/)) y 500 encuestas recolectadas a través de [Qualtrics](https://qualtricsxm8h23qkg2c.qualtrics.com/jfe/form/SV_6JeIbhOgzTszQBU).

## [RoBERTa](src/RoBERTa/)

Modelo de clasificación de comentarios basado en [RoBERTa](src/RoBERTa/). Se procesaron aproximadamente 4000 comentarios para identificar el nivel de satisfacción de los usuarios, clasificándolos en:

- Positivo
- Negativo
- Neutral

Esta clasificación permite conocer la percepción general de los pasajeros sobre el servicio.

## [Q-learning](src/Q-learning/)

Implementación de [Q-learning](src/Q-learning/) utilizando los datos procesados por [RoBERTa](src/RoBERTa/). Se construyó una función de recompensa que mide el rendimiento antes y después de aplicar la optimización, permitiendo evaluar el impacto de las mejoras propuestas en el sistema de transporte.

## [GraphSAGE](src/GraphSAGE/)

Modelo basado en grafos para analizar la opinión de los pasajeros, donde cada nodo representa opiniones individuales. Combinando los datos y las mejoras obtenidas con [Q-learning](src/Q-learning/), [GraphSAGE](src/GraphSAGE/) predice y analiza los comentarios futuros, ayudando a identificar patrones y oportunidades de optimización adicionales.

