import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            # 
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################
        dimensions = self.hparams["dimensions"]
        dimensions = [i for i in dimensions if i > latent_dim]
        dimensions.append(latent_dim)
        assert dimensions[-1] == latent_dim and dimensions[-2] > latent_dim
        assert dimensions[0] == input_size
        self.encoder = nn.Sequential()
        i=0
        for size in dimensions:
            i+=1
            modul = nn.Linear(input_size, size)
            self.encoder.add_module(f'Linear{i}', modul)
            self.encoder.add_module(f'BatchNorm{i}', nn.BatchNorm1d(size))
            self.encoder.add_module(f'Relu{i}',nn.ReLU())
            input_size = size

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None
        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################


        self.decoder = nn.Sequential()
        dimensions = [latent_dim, 64, 128, 256, 512, 28*28]
        for d in dimensions:
            if d == 28*28:
                self.decoder.add_module(f'Linear{d}',nn.Linear(latent_dim, d))
                break
            self.decoder.add_module(f'Linear{d}',nn.Linear(latent_dim, d))
            self.decoder.add_module(f'BatchNorm{d}',nn.BatchNorm1d(d))
            self.decoder.add_module(f'Relu{d}',nn.ReLU())
            latent_dim = d

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.learning_rate = hparams["learning_rate"]
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        x = self.encoder(x)
        reconstruction = self.decoder(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        # self._parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################
        loss_func = nn.MSELoss()
        self.train()
        
        self.optimizer.zero_grad()
        X = batch
        X = X.to(self.device)
        flattened_X = X.view(X.shape[0], -1)
        reconstruction = self.forward(flattened_X)
        loss = loss_func(reconstruction, flattened_X)
        loss.backward()
        self.optimizer.step()

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        loss_func = nn.MSELoss()
        self.eval()
        
        with torch.no_grad():
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            loss = loss_func(reconstruction, flattened_X)
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################
        
        latent_dim = encoder.latent_dim
        classifier_dim = hparams["classifier_dim"]
        classifier_depth = hparams["classifier_depth"]
        self.model = nn.Sequential()
        for i in range(classifier_depth):
            modul = nn.Linear(latent_dim, classifier_dim)
            self.model.add_module(f'Linear{i}',modul)
            self.model.add_module(f'BatchNorm{i}',nn.BatchNorm1d(classifier_dim))
            self.model.add_module(f'Relu{i}',nn.ReLU())
            latent_dim = classifier_dim
        self.model.add_module(f'Linear{i+1}',nn.Linear(latent_dim, hparams["num_classes"]))
        # self.model.add_module(f'Softmax{i+1}',nn.Softmax())

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
