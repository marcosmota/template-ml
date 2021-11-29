all: help
## -----------------------------------------------------------------------------------------------
## 				Synapse - Makefile Help 
## -----------------------------------------------------------------------------------------------
## Usage:
## 	make target [PARAMETER=your_parameter]
## 

run-preprocess: ## -  Roda o script de preprocessamento
	python scripts/preprocessing.py ./data

run-preprocess-train: ## -  Roda o script de preprocessamento e treino
	run-preprocess run-train

run-train: ## -  Roda o script de treino
	mlflow run . -e training

run-tunner: ## - Roda o script de hiper tuninng.
	mlflow run . -e tunner


run-evaluete: ## - Roda o script de evaluete.
	mlflow run . -e evaluete

help: ## - Mostra o help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

## 
