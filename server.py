import torch
import torch.nn as nn
from collections import OrderedDict
import time
import threading
from model import SimpleModel

# --- Configuration --- #
NUM_CLIENTS = 2
NUM_ROUNDS = 3
CLIENT_TIMEOUT = 10 # seconds

# --- Server State --- #
server_model = SimpleModel()
client_models_received = {}
current_round = 0
round_lock = threading.Lock()

def aggregate_models():
    global server_model
    if not client_models_received:
        print("No client models to aggregate.")
        return

    print("Aggregating models...")
    # Simple federated averaging
    new_state_dict = OrderedDict()
    for k in server_model.state_dict().keys():
        new_state_dict[k] = torch.zeros_like(server_model.state_dict()[k])

    for client_id, client_state_dict in client_models_received.items():
        for k in new_state_dict.keys():
            new_state_dict[k] += client_state_dict[k] / len(client_models_received)

    server_model.load_state_dict(new_state_dict)
    print("Aggregation complete.")

def start_new_round():
    global current_round, client_models_received
    with round_lock:
        current_round += 1
        client_models_received = {}
        print(f"\n--- Starting Round {current_round} ---")
        if current_round > NUM_ROUNDS:
            print("Federated learning finished.")
            return

        # In a real system, you'd send `server_model.state_dict()` to clients
        # For this example, clients will just get the initial model each round
        # or a placeholder to simulate receiving a global model.
        print("Server model ready for clients to download (simulated).")


def handle_client_update(client_id, client_state_dict):
    global client_models_received
    with round_lock:
        if current_round > NUM_ROUNDS:
            print(f"Client {client_id} update ignored, FL finished.")
            return

        if client_id in client_models_received:
            print(f"Warning: Client {client_id} already submitted for round {current_round}.")
            return

        print(f"Received update from Client {client_id} for round {current_round}.")
        client_models_received[client_id] = client_state_dict

        if len(client_models_received) == NUM_CLIENTS:
            print(f"All {NUM_CLIENTS} clients submitted for round {current_round}.")
            aggregate_models()
            start_new_round()

# Simulate server API (e.g., using Flask/FastAPI in a real app)
# For simplicity, we directly call handle_client_update here

def main():
    print("Server initialized. Waiting for clients...")
    start_new_round() # Start the first round

    # Keep server running to accept updates (simulated)
    try:
        while current_round <= NUM_ROUNDS:
            # In a real system, this would be handled by a web framework
            # For this example, the client directly calls a server function (conceptual)
            time.sleep(1) # Keep main thread alive

    except KeyboardInterrupt:
        print("Server stopped manually.")

if __name__ == '__main__':
    main()
