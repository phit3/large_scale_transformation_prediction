import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from lstp import LSTP

class LSTPController:
    @property
    def learning_rate(self):
        if 'lstp_params' in self.config:
            if 'learning_rate' in self.config['lstp_params']:
                return self.config['lstp_params']['learning_rate']
        return 1.0e-5

    @property
    def batch_size(self):
        if 'lstp_params' in self.config:
            if 'batch_size' in self.config['lstp_params']:
                return self.config['lstp_params']['batch_size']
        return 64
    
    @property
    def num_features(self):
        if 'lstp_params' in self.config:
            if 'num_features' in self.config['lstp_params']:
                return self.config['lstp_params']['num_features']
        return 256
    
    @property
    def epochs(self):
        if 'lstp_params' in self.config:
            if 'epochs' in self.config['lstp_params']:
                return self.config['lstp_params']['epochs']
        return 10000

    @property
    def max_patience(self):
        if 'lstp_params' in self.config:
            if 'patience' in self.config['lstp_params']:
                return self.config['lstp_params']['patience']
        return 30

    @property
    def num_augs(self):
        if 'lstp_params' in self.config:
            if 'num_augs' in self.config['lstp_params']:
                return self.config['lstp_params']['num_augs']
        return 8

    @property
    def num_features(self):
        if 'lstp_params' in self.config:
            if 'num_features' in self.config['lstp_params']:
                return self.config['lstp_params']['num_features']
        return 256

    @property
    def cp_name(self):
        if 'lstp_params' in self.config:
            if 'cp_name' in self.config['lstp_params']:
                return self.config['lstp_params']['cp_name']
        return 'lstp'

    @property
    def infer_steps(self):
        if 'lstp_params' in self.config:
            if 'infer_steps' in self.config['lstp_params']:
                return self.config['lstp_params']['infer_steps']
        return 100

    def __init__(self, config):
        self.config = config
        self.lstp = LSTP(batch_size=self.batch_size, num_augs=self.num_augs, num_features=self.num_features).cuda()
        self.optimizer = optim.Adam(self.lstp.parameters(), lr=self.learning_rate)
        self.patience = self.max_patience

    def do_batch(self, batch):
        inputs, targets = batch
        inputs.cuda()
        targets.cuda()
        predictions = self.lstp(inputs)
        loss = F.binary_cross_entropy(predictions, targets)
        return predictions, targets, loss

    def train(self, train_dl, valid_dl):
        self.best_vloss = np.inf
        for epoch in range(self.epochs):
            # train phase
            self.lstp.train() 
            tlosses = []
            taccs = []
            print(f'epoch {epoch + 1}/{self.epochs}')
            for i, tbatch in enumerate(train_dl):
                self.optimizer.zero_grad()
                ps, ts, tloss = self.do_batch(tbatch)
                tloss.backward()
                self.optimizer.step()
                tlosses.append(tloss.item())

                ps = ps.detach().clone()
                ps[ps >= 0.5] = 1
                ps[ps < 0.5] = 0
                acc = (ps == ts).all(1).sum() / ps.shape[0]
                taccs.append(acc)
                print(f'\rtraining {i + 1}/{len(train_dl)} - loss: {np.mean(tlosses):.5} - accuracy: {np.mean(taccs):.4}       ', end='', flush=True)
            print()
            
            # valid phase
            self.lstp.eval()
            vlosses = []
            vaccs = []
            with torch.no_grad():
                for i, vbatch in enumerate(valid_dl):
                    ps, ts, vloss = self.do_batch(vbatch)
                    vlosses.append(vloss.item())

                    ps = ps.detach().clone()
                    ps[ps >= 0.5] = 1
                    ps[ps < 0.5] = 0
                    acc = (ps == ts).all(1).sum() / ps.shape[0]
                    part_acc = (ps == ts).sum(0) / ps.shape[0]
                    vaccs.append(acc)
                    print(f'\rvalidation {i + 1}/{len(valid_dl)} - loss: {np.mean(vlosses):.5} - accuracy: {np.mean(vaccs):.4} - part accuracy {part_acc}', end='', flush=True)
            print()

            # early stopping
            if vloss < self.best_vloss:
                self.patience = self.max_patience
                # save model
                self.best_vloss = vloss
                torch.save(self.lstp.state_dict(), os.path.join('checkpoints', f'{self.cp_name}.pth'))
            elif self.patience > 0:
                self.patience -= 1
            else:
                break

    def load_cp(self):
        self.lstp.load_state_dict(torch.load(os.path.join('checkpoints', f'{self.cp_name}.pth')))
    
    def infer(self, clustering):
        with torch.no_grad():
            #self.lstp.eval()
            all_transforms = []
            cluster_idxs = list(range(clustering.num_clusters))
            for i in range(self.infer_steps):
                print(f'\rInfering {i + 1}/{self.infer_steps}', end='', flush=True)
                inputs = []
                for b in range(self.batch_size):
                    c_idx1 = np.random.choice(cluster_idxs)
                    c_idx2 = np.random.choice([cidx for cidx in cluster_idxs if cidx != c_idx1])
                    s1 = clustering.get_random_structure(c_idx1)
                    s2 = clustering.get_random_structure(c_idx2)
                    inpt = np.concatenate([s1, s2], 0)
                    inputs.append(inpt)
                # stack inputs
                inputs = np.stack(inputs)
                # infer the predicted transformation
                transform = self.lstp(torch.tensor(inputs).cuda())
                all_transforms.append(transform)
            print()
        all_transforms = torch.concatenate(all_transforms, 0).cpu().numpy()
        support = list(all_transforms.sum(0) / all_transforms.shape[0])
        support_str = ', '.join([f'{s:.2}' for s in support])
        print(f'Support: {support_str}')
        print('Visualizing support: support.png')
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.bar(range(len(support)), support)
        plt.savefig('support.png', bbox_inches='tight')
