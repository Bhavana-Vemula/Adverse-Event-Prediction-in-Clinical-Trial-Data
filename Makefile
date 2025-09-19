install:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
\tpython -m ae_predictor.etl --config configs/default.yaml

train:
\tpython -m ae_predictor.train --config configs/default.yaml

evaluate:
\tpython -m ae_predictor.evaluate --config configs/default.yaml

docker-build:
\tdocker build -t ae-predictor .

docker-train:
\tdocker run --rm -it -v $(PWD):/app ae-predictor python -m ae_predictor.train --config configs/default.yaml
