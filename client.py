import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import random
from model import SimpleModel

# In a real scenario, clients would communicate with the server via network requests
# For this simplified example, we'll import and call server functions directly
# This means server.py needs to be running or its functions available.
# DO NOT DO THIS IN A REAL DISTRIBUTED SYSTEM; USE RPC, HTTP, ETC.
import server # Simulating direct access for simplicity

# --- Client Configuration --- #
EPOCHS_PER_ROUND = 3
LEARNING_RATE = 0.01
BATCH_SIZE = 16

def get_dummy_data():
    # Simulate client's private dataset
    X = torch.randn(100, 10) # 100 samples, 10 features
    y = torch.randn(100, 1)  # 100 labels
    return torch.utils.data.TensorDataset(X, y)

def train_client_model(client_id, model, train_loader):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print(f"Client {client_id}: Starting training for {EPOCHS_PER_ROUND} epochs.")
    for epoch in range(EPOCHS_PER_ROUND):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        # print(f"Client {client_id} - Epoch {epoch+1}/{EPOCHS_PER_ROUND}, Loss: {loss.item():.4f}")
    print(f"Client {client_id}: Training finished. Final Loss: {loss.item():.4f}")
    return model.state_dict()

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument('--client_id', type=int, required=True, help='Unique ID for the client')
    args = parser.parse_args()

    client_id = args.client_id
    print(f"Client {client_id} started.")

    # Assume the client gets the initial global model from the server
    # In this example, clients start from the random initialization
    # In a real system, you'd fetch server.server_model.state_dict()
    client_model = SimpleModel()

    dataset = get_dummy_data()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    current_round_on_client = 0
    while True:
        # Simulate waiting for server to signal a new round
        # In a real FL system, this would be a long-poll or push notification
        if server.current_round > current_round_on_client:
            current_round_on_client = server.current_round
            if current_round_on_client > server.NUM_ROUNDS:
                print(f"Client {client_id}: Federated learning completed.")
                break

            print(f"Client {client_id}: Participating in round {current_round_on_client}.")
            # Simulate receiving the global model from the server
            # client_model.load_state_dict(server.server_model.state_dict())
            # For simplicity, each client just trains its current model each round

            # Train on local data
            local_state_dict = train_client_model(client_id, client_model, train_loader)

            # Send update to server
            print(f"Client {client_id}: Sending update to server.")
            server.handle_client_update(client_id, local_state_dict)

        time.sleep(random.uniform(1, 3)) # Simulate client processing time and polling

if __name__ == '__main__':
    main()
