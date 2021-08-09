help:
	@echo 'download: downloads the dataset'
	@echo 'run: runs main.py: Train + Test + Visualize'

download:
	sh download.sh

run:
	python3 -W ignore main.py
