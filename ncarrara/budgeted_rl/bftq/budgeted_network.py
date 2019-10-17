import torch

from ncarrara.utils.torch_utils import BaseModule


class BudgetedNetwork(BaseModule):
    def __init__(self, size_state, size_beta_encoder, layers,
                 activation_type="RELU",
                 reset_type="XAVIER",
                 normalize=False,
                 beta_encoder_type="LINEAR",
                 device="cpu",
                 **kwargs):
        super(BudgetedNetwork, self).__init__(activation_type, reset_type, normalize)
        sizes = layers
        self.beta_encoder_type = beta_encoder_type
        self.size_state = size_state
        self.size_beta_encoder = size_beta_encoder
        self.size_action = sizes[-1] / 2
        layers = []
        if size_beta_encoder > 1:
            if self.beta_encoder_type == "LINEAR":
                self.beta_encoder = torch.nn.Linear(1, size_beta_encoder)
            self.concat_layer = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
        else:
            module = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
            layers.append(module)
        for i in range(0, len(sizes) - 2):
            module = torch.nn.Linear(sizes[i], sizes[i + 1])
            layers.append(module)
        self.linears = torch.nn.ModuleList(layers)
        self.predict = torch.nn.Linear(sizes[-2], sizes[-1])
        self.to(device)

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std

        if self.size_beta_encoder > 1:
            beta = x[:, -1].unsqueeze(1)
            if self.beta_encoder_type == "REPEAT":
                encoded_beta = beta.repeat(1, self.size_beta_encoder)
            elif self.beta_encoder_type == "LINEAR":
                encoded_beta = self.beta_encoder(beta)
            else:
                raise "Unknown encoder type : {}".format(self.beta_encoder_type)
            state = x[:, 0:-1]
            x = torch.cat((state, encoded_beta), dim=1)
            x = self.concat_layer(x)
        elif self.size_beta_encoder == 1:
            pass
        else:
            x = x[:, 0:-1]

        for i, layer in enumerate(self.linears):
            x = self.activation(layer(x))
        x = self.predict(x)

        return x.view(x.size(0), -1)