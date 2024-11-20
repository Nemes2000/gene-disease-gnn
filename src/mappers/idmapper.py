import pandas as pd

from config import Config

class IdMapper():
    sorted_diseases = []
    sorted_genes = []

    def __init__(self, gene_file, disease_file):
        genes = pd.read_csv(gene_file, sep="\t")
        self.genes = genes["genes"].sort_values().unique()

        disieses = pd.read_csv(disease_file, sep="\t")
        diseases_filtered = disieses.groupby("diseaseId").filter(lambda x: len(x) > Config.min_disease_s_gene_number)
        self.diseases = diseases_filtered["diseaseId"].sort_values().unique()

    def diseases_idx_to_id_map(self):
        return { idx: item  for idx, item in enumerate(self.diseases)}

    def diseases_id_to_idx_map(self):
        return { item: idx  for idx, item in enumerate(self.diseases)}

    def genes_idx_to_id_map(self):
        return { idx: item  for idx, item in enumerate(self.genes)}

    def genes_id_to_idx_map(self):
        return { item: idx  for idx, item in enumerate(self.genes)}