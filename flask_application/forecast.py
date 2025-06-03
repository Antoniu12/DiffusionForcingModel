import torch

def predict_next_hour(model, context_seq, device='cpu'):
    model.eval()
    with torch.no_grad():
        x = context_seq.unsqueeze(0).to(device).float()
        y_pred_seq = model(x)
        return y_pred_seq[0, -1]

def predict_next_24h(model, context_seq, steps=24, device='cpu'):
    model.eval()
    x = context_seq.unsqueeze(0).to(device).float()
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            y_pred = model(x)
            next_step = y_pred[:, -1, :]
            predictions.append(next_step.squeeze(0))
            x = torch.cat([x[:, 1:, :], next_step.unsqueeze(1)], dim=1)

    return torch.stack(predictions)
