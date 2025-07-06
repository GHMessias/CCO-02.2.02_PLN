#!/bin/bash

# Define o timestamp com a data e hora
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Cria uma nova pasta dentro de logs com o timestamp
mkdir -p "logs/$timestamp"

mkdir -p "logs/$timestamp/graphs"

mkdir -p "logs/$timestamp/figures"

mkdir -p "logs/$timestamp/scores"

# Preciso criar a pasta das imagens

# Percorre todos os arquivos .json dentro da pasta inputs
for config_file in inputs/*.json; do
    # Verifica se o arquivo existe e é um arquivo regular
    if [ -f "$config_file" ]; then
        echo "Executando com configuração: $config_file"
        # Executa o modelo com o arquivo de configuração atual
        python3 main.py --benchmark --data "$timestamp" --config "$config_file"
    fi
done
