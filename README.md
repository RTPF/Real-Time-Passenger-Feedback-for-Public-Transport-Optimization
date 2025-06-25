# Real-Time Passenger Feedback for Public Transport Optimization

This project seeks to optimize public transportation by systematically analyzing passenger feedback and comments. It integrates natural language processing (NLP), reinforcement learning (RL), and graph analysis to enhance user experience and system efficiency, utilizing user feedback as the foundation for informed decision-making.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [RoBERTa](#roberta)
- [Independet Q-learning](#independet-q-learning)
- [GraphSAGE](#graphsage)

## Overview

The system integrates sentiment classification models, reinforcement learning, and graph analysis to process and leverage feedback from public transport users. The general workflow is as follows:

1. Data and comment collection from users.
2. Classification of comments utilizing [RoBERTa](src/RoBERTa/) (positive, negative, neutral).
3. Reward and policy optimization through [Independet Q-learning](src/Q-learning/).
4. Analysis of user relationships and patterns using [GraphSAGE](src/GraphSAGE/).

## Dataset

The [`data`](data) directory contains the dataset employed in this project, compiled from information provided by the **Coordinación General de Movilidad** ([CMOV](https://www.aguascalientes.gob.mx/cmov/)) and 500 surveys collected via [Qualtrics](https://qualtricsxm8h23qkg2c.qualtrics.com/jfe/form/SV_6JeIbhOgzTszQBU).

## [RoBERTa](src/RoBERTa/)

A comment classification model based on [RoBERTa](src/RoBERTa/). Approximately 4,000 comments were processed to determine user satisfaction levels, categorizing them as:

- Positive
- Negative
- Neutral

This classification provides valuable insights into the general perception of passengers regarding the service.

## [Independet Q-learning](src/Q-learning/)

[Independet Q-learning](src/Q-learning/) is implemented using the data processed by [RoBERTa](src/RoBERTa/). A reward function was designed to assess performance before and after optimization, thereby enabling the evaluation of the impact of proposed improvements within the transport system.

## [GraphSAGE](src/GraphSAGE/)

A graph-based model designed to analyze passenger opinions, where each node represents an individual opinion. By integrating the data and improvements obtained through [Independet Q-learning](src/Q-learning/), [GraphSAGE](src/GraphSAGE/) predicts and analyzes future comments, facilitating the identification of patterns and additional opportunities for optimization.
