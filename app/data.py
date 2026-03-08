import numpy as np
from pathlib import Path


CURRENT_PATH = Path(__file__).parent
DATA_PATH = CURRENT_PATH / '..' / 'data'


def load_quarterback():
    """
    Carrega o conjunto de dados concernentes a pontos feitos pelo principal jogador (quarterback)
    de ataque do futebol americano durante o ano de 2004 da Liga Americana de Futebol (fonte:
    The Sports Network).
    """
    filepath = DATA_PATH / 'qb_2004.csv'
    x = []
    y = []
    with open(filepath) as f:
        for line in f.readlines()[1:]:
            *_, yards, scores = line.replace('\n', '').split(',')
            x.append(yards)
            y.append(scores)

    return np.array(x, dtype=float), np.array(y, dtype=float)


def load_radiation():
    """
    Carrega os dados de radiação
    Retorno: x, y
    """
    data = np.loadtxt(DATA_PATH / 'dose_radiacao_expandido.csv', skiprows=1, delimiter=',')
    return data[:, 0:3], data[:, 3]


if __name__ == '__main__':
    print(load_radiation())
