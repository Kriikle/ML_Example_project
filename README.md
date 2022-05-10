This project build uses [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset.
## Usage
1. Clone this repository to your machine.
2. Download  [Forest Cover](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.7.5 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. To create pandas profiling report use the following command:
```sh
poetry run eda
```
6. Run train with the following command:
```sh
poetry run train 
```
