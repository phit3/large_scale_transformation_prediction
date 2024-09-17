import numpy as np
import matplotlib.pyplot as plt


class TimeConstrainedClustering:
    @property
    def num_clusters(self):
        return len(self.peaks) + 1

    @property
    def c_min(self):
        if 'data_params' in self.config:
            if 'c_min' in self.config['data_params']:
                return self.config['data_params']['c_min']
        return 10

    @property
    def c_max(self):
        if 'data_params' in self.config:
            if 'c_max' in self.config['data_params']:
                return self.config['data_params']['c_max']
        return 100

    @property
    def c_shift(self):
        if 'data_params' in self.config:
            if 'c_shift' in self.config['data_params']:
                return self.config['data_params']['c_shift']
        return 1

    @property
    def c_size(self):
        if 'data_params' in self.config:
            if 'c_size' in self.config['data_params']:
                return self.config['data_params']['c_size']
        return 100

    def __init__(self, data, config):
        self.data = data
        self.config = config

    def check_local_groups(self, idx):
        grp1 = self.data[max(0, idx - self.c_max): idx]        
        grp2 = self.data[idx: idx + self.c_max]
        grp1_mean = grp1.mean(0)
        grp2_mean = grp2.mean(0)
        diff = np.mean((grp1_mean - grp2_mean)**2)
        return diff

    def get_cluster(self, idx):
        idx = idx % self.num_clusters
        if idx == len(self.peaks):
            start = self.peaks[-1][0]
            end = self.data.shape[0]
        elif idx == 0:
            start = 0
            end = self.peaks[0][0]
        else:
            start = self.peaks[idx - 1][0]
            end = self.peaks[idx][0]
        start = int(start)
        end = int(end)
        return self.data[start:end]

    def get_random_structure(self, idx):
        idx = idx % self.num_clusters
        cluster = self.get_cluster(idx)
        if cluster.shape[0] < self.c_size:
            print(f'[WARN] Cluster size {cluster.shape[0]} is too small for c_size {self.c_size}. Returning mean of the cluster')
            return cluster.mean(0)
        start = np.random.randint(0, cluster.shape[0] - self.c_size)
        return cluster[start:start + self.c_size].mean(0)

    def get_peaks(self):
        peaks = []
        for i in range(1, len(self.points) - 1):
            pos, diff = self.points[i]
            #diffs = [p[1] for p in self.points]
            diffs = self.points[:, 1]
            left_limit = max(0, i - int(self.c_min / self.c_shift / 2))
            right_limit = min(len(diffs), i + int(np.ceil(self.c_min / self.c_shift / 2)))
            neighborhood = np.array(list(diffs[left_limit: i]) + list(diffs[i + 1: right_limit]))
            #neighborhood = np.array(diffs[i - int(self.c_min / self.c_shift / 2): i + int(np.ceil(self.c_min / self.c_shift / 2))])
            left_diff = diffs[i - 1]
            right_diff = diffs[i + 1]
            
            max_prop = False if len(neighborhood) == 0 else diff > neighborhood.max()
            if diff > left_diff and diff > right_diff and max_prop:
                peaks.append((pos, diff))

        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        real_peaks = []
        cuts = []
        for p, d in peaks:
            if len(cuts) == 0:
                real_peaks.append((p, d))
                cuts.append(p)
                continue 
            min_dist = abs(np.array(cuts) - p).min()
            if min_dist > self.c_min:
                real_peaks.append((p, d))
                cuts.append(p)
        real_peaks = np.array(sorted(real_peaks, key=lambda x: x[0]))
        return real_peaks

    def fit(self):
        diffs = []
        poss = []
        for split_pos in range(self.c_min, self.data.shape[0] - self.c_min, self.c_shift):
            max_pos = self.data.shape[0] - self.c_min 
            print(f'\r{split_pos}/{max_pos}', end='', flush=True)
            diff = self.check_local_groups(split_pos)
            poss.append(split_pos)
            diffs.append(diff)
        print()

        self.points = np.array(list(zip(poss, diffs)))
        self.peaks = self.get_peaks()

        # plot the peaks
        print('Visualizing the peaks: peaks.png')
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.plot(self.points[:, 0], self.points[:, 1])
        axs.scatter(self.peaks[:, 0], self.peaks[:, 1], c='r')
        axs.set_xlim(0, self.data.shape[0])
        plt.savefig('peaks.png', bbox_inches='tight')

