from Grid import Grid
from sklearn.datasets import load_diabetes

import pandas as pd

X, y = load_diabetes(True)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

gr = Grid(X, y)

gr.grid_2d(1, 15, 15, 1, 50, 50)

gr.grid_d('lr', 0.1, 1, 30)

gr.grid_2d(1, 3, 3, 1, 3, 3)

# gr.grid_d('lr', 0.1, 1, 30)
