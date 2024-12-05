import time
import torch
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  train_losses, test_losses, train_accs, test_accs = [], [], [], []
  since_total = time.time()

  for epoch in tqdm.tqdm(range(num_epochs)):
    # set the model to train mode
    model.train()

    since = time.time()
    running_loss = 0.0
    running_correct = 0.0

    # iterate over the DataLoader batches
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # FORWARD
        outputs = model(inputs)
        # unsqueeze b/c outputs in shape (batch_size, 1)
        # loss = criterion(outputs, labels.float().unsqueeze(1))
        loss = criterion(outputs.squeeze(1), labels.float())
        # get predicted label (0/1)
        # probs = torch.sigmoid(outputs)
        predicted = (outputs.detach() >= 0).long().squeeze(1)

        # BACKWARD and OPTIMIZE
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_correct += torch.sum(predicted == labels.data).item()

    epoch_duration = time.time() - since
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_correct / len(train_loader.dataset)

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # EVALUATE model on the test set
    test_loss, test_acc = eval_model(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"\nEpoch {epoch}/{num_epochs - 1} Duration: {epoch_duration:.3f} s Train Loss: {epoch_loss:.4f} Train Accuracy: {epoch_acc:.4f} Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}")

  total_time = time.time() - since_total
  print(f"\nTotal Training Time: {total_time:.4f} s")
  print("Finished Training.")
  return train_losses, train_accs, test_losses, test_accs


def eval_model(model, test_loader, criterion):
  # switch the model to eval mode
  model.eval()
  running_loss = 0.0
  running_correct = 0

  for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    # don't have to update gradient during evaluation
    with torch.no_grad():
      outputs = model(inputs)
      # probs = torch.sigmoid(outputs)
      predicted = (outputs.detach() >= 0).long().squeeze(1)
      loss = criterion(outputs.squeeze(1), labels.float())

    running_loss += loss.item()
    running_correct += torch.sum(predicted == labels.data).item()

  epoch_loss = running_loss / len(test_loader.dataset)
  epoch_acc = running_correct / len(test_loader.dataset)

  return epoch_loss, epoch_acc

