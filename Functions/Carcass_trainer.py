# IMPORTS
from matplotlib import pyplot as plt 
import gc
from tqdm import tqdm

class Trainer():

  def __init__(self, model, criterion, optimizer, dataset, data_loader,device ,lr=0.01,epochs=10, batch_size=3,class_weights=None):
    self.lr=lr
    self.batch_size=batch_size
    self.device=device

    self.model=model.to(self.device)
    if class_weights is not None:
      self.criterion=criterion(weight=class_weights.to(self.device))
    else:
      self.criterion=criterion   # arg in the main code
    self.optimizer=optimizer(self.model.parameters(),lr=self.lr)
    self.dataset=dataset
    self.data_loader=data_loader(self.dataset, batch_size=self.batch_size)

  def forward(self, X):
    return self.model(X)

  def train(self, epochs=10, plot=True):
    step_losses = []
    epoch_losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X, Y in tqdm(self.data_loader, total=len(self.data_loader), leave=False):
            X, Y = X.to(self.device), Y.to(self.device)
            self.optimizer.zero_grad()
            Y_pred = self.model(X)
            Y_pred = Y_pred.squeeze(1)
            loss = self.criterion(Y_pred, Y.long())
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())

            del X, Y, Y_pred, loss
            gc.collect()
        epoch_losses.append(epoch_loss / len(self.data_loader))

    if plot:
      fig, axes = plt.subplots(1, 2, figsize=(10, 5))
      axes[0].plot(step_losses)
      axes[1].plot(epoch_losses)
    else:
      return step_losses, epoch_losses


