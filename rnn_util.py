import time
import torch
import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, X_train, y_train, X_test, y_test, criterion, optimizer, num_epochs=10):
  model.to(device)

  train_losses, test_losses, train_accs, test_accs = [], [], [], []
  since_total = time.time()

  for epoch in tqdm.tqdm(range(num_epochs)):
    # Shuffle X_train and y_train
    combined = list(zip(X_train, y_train))
    np.random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # set the model to train mode
    model.train()

    since = time.time()
    running_loss = 0.0
    running_correct = 0.0

    # iterate over each data point
    for i in range(len(X_train)):
        inputs, label = torch.tensor(X_train[i]), torch.tensor(y_train[i])
        label = label.view(1, -1)  # [1,1] shape
        inputs, label = inputs.to(device), label.to(device)
        optimizer.zero_grad()

        # FORWARD
        output = model(inputs)
        # unsqueeze b/c outputs in shape (batch_size, 1)
        loss = criterion(output, label.float())
        # get predicted label (0/1)
        probs = torch.sigmoid(output)
        predicted = (probs >= 0.5).int()
        # print("predicted: ", predicted)  # tensor([[1]])
        # print("label: ", label) # tensor([[1]])

        # BACKWARD and OPTIMIZE
        # zero the parameter gradients
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_correct += (predicted == label).sum().item()

    epoch_duration = time.time() - since
    epoch_loss = running_loss / len(X_train)
    epoch_acc = running_correct / len(X_train)

    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # EVALUATE model on the test set
    test_loss, test_acc = eval_model(model, X_test, y_test, criterion)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"\nEpoch {epoch}/{num_epochs - 1} Duration: {epoch_duration:.3f} s Train Loss: {epoch_loss:.4f} Train Accuracy: {epoch_acc:.4f} Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}")

  total_time = time.time() - since_total
  print(f"\nTotal Training Time: {total_time:.4f} s")
  print("Finished Training.")
  return train_losses, train_accs, test_losses, test_accs


def eval_model(model, X_test, y_test, criterion):
    # switch the model to eval mode
    model.eval()
    running_loss = 0.0
    running_correct = 0

    for i in range(len(X_test)):
        inputs, label = torch.tensor(X_test[i]), torch.tensor(y_test[i])
        label = label.view(1, -1)  # [1,1] shape
        inputs, label = inputs.to(device), label.to(device)

        # don't have to update gradient during evaluation
        with torch.no_grad():
            output = model(inputs)
            loss = criterion(output, label.float())
            probs = torch.sigmoid(output)
            predicted = (probs >= 0.5).int()

        running_loss += loss.item()
        running_correct += (predicted == label).sum().item()

    epoch_loss = running_loss / len(X_test)
    epoch_acc = running_correct / len(X_test)

    return epoch_loss, epoch_acc

