import numpy as np
from collections import Counter
import pandas as pd

class CLOPE:
    def __init__(self, k, r):
        self.k = k
        self.r = r
        self.cluster_rows = {i: [] for i in range(k)}
        self.cluster_histogram = {i: {} for i in range(k)}
        self.cluster_items_count = {i: 0 for i in range(k)}

    @staticmethod
    def preprocess_data(df):
        for colnum in range(len(df.columns)):
            df.iloc[:, colnum] = df.iloc[:, colnum].astype(str)
            df.iloc[:, colnum] = df.iloc[:, colnum].apply(lambda x: f"{colnum}|{x}")
        return df.to_numpy()

    @staticmethod
    def add_to_histogram(cluster_histogram, row_vector):
        cluster_counter = Counter(cluster_histogram)
        row_counter = Counter(row_vector)
        updated_counter = cluster_counter + row_counter
        return dict(updated_counter)

    @staticmethod
    def remove_from_histogram(cluster_histogram, row_vector):
        cluster_counter = Counter(cluster_histogram)
        row_counter = Counter(row_vector)
        updated_counter = cluster_counter - row_counter
        return dict(updated_counter)

    def add_transaction(self, cluster_id, row_index, row_vector):
        self.cluster_rows[cluster_id].append(row_index)
        self.cluster_histogram[cluster_id] = self.add_to_histogram(self.cluster_histogram[cluster_id], row_vector)
        self.cluster_items_count[cluster_id] += len(row_vector)

    def remove_transaction(self, cluster_id, row_index, row_vector):
        self.cluster_rows[cluster_id].remove(row_index)
        self.cluster_histogram[cluster_id] = self.remove_from_histogram(self.cluster_histogram[cluster_id], row_vector)
        self.cluster_items_count[cluster_id] -= len(row_vector)

    def delta_add(self, C_S, C_W, C_N, t_item_count, t_dW):
        S_new = C_S + t_item_count
        W_new = C_W + t_dW
        return S_new * (C_N + 1) / (W_new ** self.r) - C_S * C_N / (C_W ** self.r)

    def assign_to_clusters(self, data):
        for row_index, row_vector in enumerate(data):
            empty_clusters = [cluster_id for cluster_id, rows in self.cluster_rows.items() if not rows]

            if empty_clusters:
                best_cluster = None
                C_S = len(row_vector)
                C_W = len(set(row_vector))
                max_profit = (C_S / C_W**self.r)
                best_cluster = empty_clusters[0]

                for cluster_id in self.cluster_rows:
                    if len(self.cluster_rows[cluster_id]) == 0:
                        continue

                    C_S = self.cluster_items_count[cluster_id]
                    C_W = len(self.cluster_histogram[cluster_id]) if C_S > 0 else 0
                    C_N = len(self.cluster_rows[cluster_id])

                    t_item_count = len(row_vector)
                    t_dW = len(set(row_vector) - set(self.cluster_histogram[cluster_id].keys()))

                    profit = self.delta_add(C_S, C_W, C_N, t_item_count, t_dW)

                    if profit > max_profit:
                        max_profit = profit
                        best_cluster = cluster_id

                self.add_transaction(best_cluster, row_index, row_vector)

            else:
                best_cluster = None
                max_profit = float('-inf')

                for cluster_id in self.cluster_rows:
                    C_S = self.cluster_items_count[cluster_id]
                    C_W = len(self.cluster_histogram[cluster_id]) if C_S > 0 else 0
                    C_N = len(self.cluster_rows[cluster_id])

                    t_item_count = len(row_vector)
                    t_dW = len(set(row_vector) - set(self.cluster_histogram[cluster_id].keys()))

                    profit = self.delta_add(C_S, C_W, C_N, t_item_count, t_dW)

                    if profit > max_profit:
                        max_profit = profit
                        best_cluster = cluster_id

                if row_index == 0:
                    best_cluster = 0

                if best_cluster is not None:
                    self.add_transaction(best_cluster, row_index, row_vector)

    def refine_clusters(self, data):
        for cluster_id in range(self.k):
            if len(self.cluster_rows[cluster_id]) > 0:
                for row_index in self.cluster_rows[cluster_id][:]:
                    row_vector = data[row_index]

                    best_cluster = None
                    max_profit = float('-inf')

                    C_S = self.cluster_items_count[cluster_id]
                    C_W = len(self.cluster_histogram[cluster_id]) if C_S > 0 else 0
                    C_N = len(self.cluster_rows[cluster_id])

                    t_item_count = len(row_vector)
                    t_dW = len(set(row_vector) - set(self.cluster_histogram[cluster_id].keys()))

                    profit = self.delta_add(C_S, C_W, C_N, t_item_count, t_dW)

                    if profit > max_profit:
                        max_profit = profit
                        best_cluster = cluster_id

                    for other_cluster_id in range(self.k):
                        if other_cluster_id != cluster_id and len(self.cluster_rows[other_cluster_id]) > 0:
                            C_S = self.cluster_items_count[other_cluster_id]
                            C_W = len(self.cluster_histogram[other_cluster_id]) if C_S > 0 else 0
                            C_N = len(self.cluster_rows[other_cluster_id])

                            t_item_count = len(row_vector)
                            t_dW = len(set(row_vector) - set(self.cluster_histogram[other_cluster_id].keys()))

                            profit = self.delta_add(C_S, C_W, C_N, t_item_count, t_dW)

                            if profit > max_profit:
                                max_profit = profit
                                best_cluster = other_cluster_id

                    if best_cluster is not None and max_profit > 0 and best_cluster != cluster_id:
                        self.remove_transaction(cluster_id, row_index, row_vector)
                        self.add_transaction(best_cluster, row_index, row_vector)

    def fit(self, df):
        data = self.preprocess_data(df)
        self.assign_to_clusters(data)
        self.refine_clusters(data)

    def get_clusters(self):
        return {
            "rows": self.cluster_rows,
            "histograms": self.cluster_histogram,
            "items_count": self.cluster_items_count,
        }
