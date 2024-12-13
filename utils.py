import torch

def save_model(model, path):
    """
    Save the model to the specified path.
    """
    torch.save(model.state_dict(), path)


def load_model(path):
    """
    Load a model from the specified path.
    """
    model = torch.load(path)
    return model
