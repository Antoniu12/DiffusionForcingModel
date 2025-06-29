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

def autoregressive_next_24(model, context_seq, steps=24, device='cpu'):
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

def teacher_forcing_next_24(model, context_seq, ground_truth_seq, steps=24, device='cpu'):
    model.eval()
    x = context_seq.unsqueeze(0).to(device).float()
    ground_truth_seq = ground_truth_seq.to(device).float()
    predictions = []

    with torch.no_grad():
        for i in range(steps):
            y_pred = model(x)
            next_step = y_pred[:, -1, :]
            predictions.append(next_step.squeeze(0))
            true_next = ground_truth_seq[i].unsqueeze(0).unsqueeze(0)
            x = torch.cat([x[:, 1:, :], true_next], dim=1)

    return torch.stack(predictions)
