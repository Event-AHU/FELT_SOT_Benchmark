import torch


def patch2token(patch):
    B, C, H, W = patch.shape
    token = patch.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
    return token


def token2patch(token):
    B, N, C = token.shape
    if N == 64 or N == 128:
        H = W = 8
    elif N == 256 or N == 512:
        H = W = 16
    elif N == 144:
        H = W = 12
    elif N ==1024:
        H = W = 32
    else:
        raise

    patch = token.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
    return patch


if __name__ == '__main__':
    a = torch.rand(1, 256, 768)
    b = token2patch(a)
    print(b.shape)
