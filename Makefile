help:
	@echo 'download: downloads the dataset'
	@echo 'train: runs train.py to train the model'

download:
	sh download.sh

train:
	python3 -W ignore train.py
