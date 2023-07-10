import torch

def normalize(x):
    t_d = torch.median(x)
    s_d = torch.mean(torch.abs(x - t_d))
    return (x - t_d) / (s_d+1e-9)

def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.tensor([10.],device=x.device))

def L2_norm(x,epsilon=1e-9):
    if len(x.shape) == 2:
        return (torch.sum(x**2, dim=-1) + epsilon).mean()+1e-9
    else:
        return (x**2 + epsilon).mean()+1e-9

def L1_norm(x, M=None, epsilon=1e-9):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + epsilon) / x.shape[-1]

    
def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-19)) / x.shape[0]

def positional_encoding(t, num_encodings=10):
    """
    Encodes time t with a positional encoding.

    :param t: Time tensor (shape: [batch_size, 1])
    :param num_encodings: The number of positional encodings to generate
    :return: Positionally encoded time (shape: [batch_size, num_encodings])
    """
    # Generate a range of frequencies
    frequencies = 2.0 ** torch.linspace(0.0, num_encodings - 1, num_encodings).unsqueeze(0)
    # frequencies = frequencies.unsqueeze(0)
    # assert False, f"frequencies.shape {frequencies.shape} t.shape {t.shape}"
    # Encode t with these frequencies
    try:
        encoded_t = t * frequencies.to(t.device)
    except:
        assert False, f"t.shape: {t.shape}, frequencies.shape: {frequencies.shape}"
    # encoded_t = t * frequencies.to(t.device)

    # Apply sin and cos to generate positional encodings
    encoded_t = torch.cat([encoded_t.sin(), encoded_t.cos()], dim=-1)

    return encoded_t

