import torch
import json
import dyadic_dit_vq.model.vqvae_downsampled as vqvae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = json.load(open("/home/peng/dyadic_motion/dyadic_dit/dyadic_dit_vq/ckpt/args.json"))
model = vqvae.TemporalVertexCodecDownsampled(
    n_vertices=args["nb_joints"],
    latent_dim=args["output_emb_width"],
    categories=args["code_dim"],
    residual_depth=args["depth"],
    downsample_factor=args["downsample_factor"],
).to(device)

model.load_state_dict(torch.load("/home/peng/dyadic_motion/dyadic_dit/dyadic_dit_vq/ckpt/net_best.pth", weights_only=False)['net'])
model.eval()
model.to(device)

dummy_motion = torch.randn(1, 96, 258).to(device)


def encode_motion(motion):
    """
    Encode motion to discrete tokens
    motion: [batch, seq_len, motion_dim]
    return: [batch, seq_len//downsample_factor, latent_dim]
    """
    encoded = model.encode(motion)
    return encoded

def decode_motion(encoded):
    """
    Decode discrete tokens to motion
    encoded: [batch, seq_len//downsample_factor, latent_dim]
    return: [batch, seq_len, motion_dim]
    """
    decoded = model.decode(encoded)
    return decoded

def get_codebook_representation(encoded):
    """
    Get codebook representation
    encoded: [batch, seq_len//downsample_factor, latent_dim]
    return: [batch, seq_len//downsample_factor, codebook_dim]
    """
    reformat = encoded.dim() > 2
    if reformat:
        B, T_down, _ = encoded.shape
        encoded = encoded.reshape((-1, model.residual_depth))
    codebook_representation = model.quantizer.decode(encoded)
    if reformat:
        codebook_representation = codebook_representation.reshape((B,T_down,-1))
    return codebook_representation


print(encode_motion(dummy_motion).shape) # [1, 24, 64]
print(decode_motion(encode_motion(dummy_motion)).shape) # [1, 96, 258]  
print(get_codebook_representation(encode_motion(dummy_motion)).shape) # [1, 24, 64]