PYTHON ?= python

.PHONY: prepare-data train-model evaluate-model full-pipeline clean

prepare-data:
	$(PYTHON) -m src.data.prepare

train-model: prepare-data
	$(PYTHON) -m src.models.train

evaluate-model: train-model
	$(PYTHON) -m src.models.evaluate

full-pipeline: evaluate-model

clean:
	rm -rf data/processed models/*.joblib reports/*.json reports/*.csv
