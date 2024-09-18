import torch
from torchvision.models import resnet50


class LSTP(torch.nn.Module):
    def __init__(self, batch_size, num_augs, num_features):
        super().__init__()
        self.batch_size = batch_size
        self.num_augs = num_augs
        self.num_features = num_features
        self.drop_p = 0.1
        self.combine_kernel_size = 3

        res = resnet50(pretrained=False)
        res_layers = list(res.children())
        res_layers = res_layers[:-4]

        res_layers[0] = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.state_size = list(list(res_layers[-1].children())[-1].children())[-3].out_channels

        self.conv_model = torch.nn.Sequential(*res_layers)

        self.conv_combine_states = torch.nn.Conv2d(self.state_size * 2, self.state_size, kernel_size=self.combine_kernel_size)
        self.aap = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.drop1 = torch.nn.Dropout(self.drop_p)

        self.exi_lin1 = torch.nn.Linear(3 * self.state_size, 2 * self.num_features)
        self.exi_lin2 = torch.nn.Linear(2 * self.num_features, self.num_features)
        self.exi_lin3 = torch.nn.Linear(self.num_features, self.num_augs)

    def forward(self, inputs) -> torch.Tensor:
        super1 = inputs[:, :1, :, :]
        super2 = inputs[:, 1:, :, :]
        sf1 = self.conv_model(super1)
        sf2 = self.conv_model(super2)
        sfb = self.conv_combine_states(torch.cat((sf1, sf2), dim=1))
        sfb = torch.relu(sfb)
        f1 = self.aap(sf1).squeeze()
        f2 = self.aap(sf2).squeeze()
        fb = self.aap(sfb).squeeze()

        fs = torch.cat((f1, f2, fb), dim=1)
        exi_fs = torch.relu(self.exi_lin1(fs))
        exi_fs = self.drop1(exi_fs)
        exi_fs = torch.relu(self.exi_lin2(exi_fs))
        transform = torch.sigmoid(self.exi_lin3(exi_fs))
        return transform


