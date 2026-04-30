import numpy as np
import pandas as pd


### не используется нигде, это исскуственные данные

def generate_data():
    np.random.seed(42)

    cpu = np.random.normal(50, 10, 1000)
    memory = np.random.normal(60, 10, 1000)
    
    data = pd.DataFrame({
        "cpu": cpu,
        "memory": memory
    })

    data.loc[900:920, "cpu"] = 95
    data.loc[930:940, "memory"] = 10

    return data