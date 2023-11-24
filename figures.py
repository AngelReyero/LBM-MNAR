import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import yaml 


def groupes_politiques(row_classes, deputes): 
    dep = pd.DataFrame(deputes)['groupe']
    groups = dep.unique()
    count  = np.zeros_like(groups)
    group_per_class = []

    for nq in range(len(deputes)): 
        vec = []
        for i, c in enumerate(row_classes): 
            if c==nq: 
                count[np.where(dep[i]==groups)[0]] += 1
                if dep[i] not in vec: 
                    vec.append(dep[i])
        if vec != []: 
            group_per_class.append(vec)
            
    return group_per_class, pd.DataFrame([count], columns=pd.Index(groups)) 