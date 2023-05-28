import torch

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

