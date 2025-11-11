
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from losses import get_loss

def train_model(model, dataset, config, device):
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 損失関数 ---
    criterion = get_loss(config["loss"])

    # --- optimizer ---
    opt_cfg = config["training"]
    if opt_cfg["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt_cfg["learning_rate"], weight_decay=opt_cfg.get("weight_decay", 0.0))
    elif opt_cfg["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt_cfg["learning_rate"], momentum=opt_cfg.get("momentum", 0.9))
    else:
        raise ValueError(f"Unsupported optimizer: {opt_cfg['optimizer']}")

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
