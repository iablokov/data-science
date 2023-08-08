import pandas as pd
from typing import Optional
from collections import defaultdict

class DatasetProfiler:
    
    def __init__(self):
        
        self.datasets        : defaultdict[dict] = defaultdict(dict)

        # dataframe and metadata columns (all and shared)
        self.features_all    : pd.Index = pd.Index([])
        self.features_shared : pd.Index = pd.Index([])
        self.labels_all      : pd.Index = pd.Index([])
        self.labels_shared   : pd.Index = pd.Index([])
    
    def add_dataset(self, name     : str, 
                          dataset  : pd.DataFrame, 
                          metadata : Optional[pd.DataFrame] = None) -> None:
        
        if metadata.index != dataset.index:
            raise ValueError("Index of dataset and metadata must be the same")
        
        # metadata["dataset"] = name ???
        self.datasets[name] = { "dataset" : dataset, "metadata" : metadata }
        self._update_features_and_labels(name)

    def _update_features_and_labels(self, name: str) -> None:
        
        self.features_all    = self.features_all.union(self.datasets[name]["dataset"].columns)
        self.features_shared = self.features_shared.intersection(self.datasets[name]["dataset"].columns)    

        self.labels_all      = self.labels_all.union(self.datasets[name]["metadata"].columns)
        self.labels_shared   = self.labels_shared.intersection(self.datasets[name]["metadata"].columns)

