import torch
from sympy import total_degree
from torch import nn


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # torch.randn_like(tensor) creates noise with the same shape/device as the image
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Residual Network
class customResnet(nn.Module):
    def __init__(self, path_layers=[], shorting_layers=None):
        super().__init__()
        self.path_layers = nn.Sequential(*path_layers)
        self.shorting = nn.Sequential(*shorting_layers) if shorting_layers else nn.Identity()

    def forward(self, x):
        return self.path_layers(x) + self.shorting(x)


# creating a simple network framework to train models
class piron(nn.Module):
    def __init__(self):
        # initial initialization
        super().__init__()

        # initialising a layer and activations stack ( a list )
        self.layer_stack_list = []

        # creating empty layer stack object
        self.layer_stack = None

        # this will become the dataloader object
        self.data = None

        # the loss function
        self.loss_fn = None

        # the optimizer object
        self.optimizer = None

    def add_layer(self, layer_object):
        # adding layers to the stack
        self.layer_stack_list.append(layer_object)

    def initialise_sequential(self):
        # creating the layer stack
        self.layer_stack = nn.Sequential(*self.layer_stack_list)

    def load_dataloader(self, dataloader):
        # loading the data loader object
        self.data = dataloader

    def load_loss(self, loss):
        # loading the loss function
        self.loss_fn = loss

    def load_optimizer(self, optimizer):
        # loading the optimizer object. Note that to create the optimizer function , the user will first have to call model.parameteres()
        self.optimizer = optimizer

    def forward(self, x):
        # forward function : compulsory to be defined
        if self.layer_stack is None:
            raise NotImplementedError('Sequential object not initialised. Use initialise_sequential to initialise')
        logits = self.layer_stack(x)
        return logits

    def epoch_run(self, epoch=100, tol_loss=0):
        # checks if all things have been initialised
        if not self.data or not self.loss_fn or not self.optimizer:
            raise RuntimeError("Data, Loss, or Optimizer not loaded!")

        # Determine device dynamically based on model parameters
        device = next(self.parameters()).device
        print(f'Model training device : {device}')
        # start the process
        size = len(self.data.dataset)
        self.train()
        for i in range(epoch):
            total_loss = 0
            for batch, (x, y) in enumerate(self.data):
                # Move tensors to the models device
                x, y = x.to(device), y.to(device)

                pred = self(x)
                loss = self.loss_fn(pred, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
            print(f'\rEpoch : {i + 1}  || Loss : {total_loss}', end='', flush=True)
            if total_loss <= tol_loss:
                print('Tolerance Loss reached . Stopping training')
                break
        print('')

    # prediction
    def predict(self, x):
        self.eval()
        device = next(self.parameters()).device

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Move input tensor to device
        x = x.to(device)

        with torch.no_grad():
            logits = self(x)

        return logits

    # singleton predict
    def predict_singleton(self, x):
        self.eval()
        device = next(self.parameters()).device

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)  # Standardized to float32 for consistency

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        # Move input tensor to device
        x = x.to(device)

        with torch.no_grad():
            logits = self(x)

        return logits

    # function to save model
    def save_model(self, filename):
        state_dict = {
            'architecture': self.layer_stack_list,
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_type': self.optimizer.__class__.__name__,
            'loss': self.loss_fn
        }
        torch.save(state_dict, filename)
        print(f'Saved model to {filename}')

    @staticmethod
    def load_model(filename):
        # Dynamically determine the best device to map the tensors to
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        # creating empty model
        x = piron()

        # loading the state dictionary with map_location to avoid GPU-to-CPU errors
        state_dict = torch.load(filename, weights_only=False, map_location=device)

        # defining the layer stack list , then initialising to create the layer objects
        x.layer_stack_list = state_dict['architecture']
        x.initialise_sequential()

        # loading the weights
        x.load_state_dict(state_dict['model'])

        # Move the entire model to the selected device
        x.to(device)

        import torch.optim as optim

        # loading the old optimizer
        optimizer_class = getattr(optim, state_dict['optimizer_type'])
        optimizer = optimizer_class(x.parameters(), lr=1)

        optimizer.load_state_dict(state_dict['optimizer'])

        x.load_optimizer(optimizer)

        # defining loss function
        x.load_loss(state_dict['loss'])

        return x