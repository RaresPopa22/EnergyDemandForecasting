# Energy Demand Forecasting

This project aims to predict next hour energy demand for a specific country, Romania. As models I used LSTM and a baselines naive seasonal forecasting model. The dataset was fetched from https://www.entsoe.eu/data/power-stats/ using a custom data_fetcher which grabbed hourly data from 2019 to 2025. The dataset is enhanced with meteo data for every hour, for the biggest city in Romania, Bucharest, which can be a good proxy for energy consumption based on the temperature conditions.

### Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Usage](#usage)
* [Configuration](#configuration)
* [Model Performance](#model-performance)


### Introduction

Using pytorch LSTM model 