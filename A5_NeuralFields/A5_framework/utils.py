import torch

def bilinear_interp(grid, pts, res, grid_type = 'dense'):
    #grids 2D tensor of size (res ** 3, feat_dim)
    #pts 2D tensor of size (num_pts, 3)
    PRIMES = [1, 265443567, 805459861]
    #resize grid
    if grid_type == 'dense':
        grid = grid.reshape(res, res, res, -1)

    xs = (pts[:, 0] + 1) * 0.5 * (res - 1)
    ys = (pts[:, 1] + 1) * 0.5 * (res - 1)
    zs = (pts[:, 2] + 1) * 0.5 * (res - 1)

    x0 = torch.floor(torch.clip(xs, 0, res - 1 - 1e-5)).long()
    y0 = torch.floor(torch.clip(ys, 0, res - 1 - 1e-5)).long()
    z0 = torch.floor(torch.clip(zs, 0, res - 1 - 1e-5)).long()

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    #get weights
    w1 = ((x1 - xs) * (y1 - ys) * (z1 - zs)).unsqueeze(1).cuda()
    w2 = ((xs - x0) * (y1 - ys) * (z1 - zs)).unsqueeze(1).cuda()
    w3 = ((x1 - xs) * (ys - y0) * (z1 - zs)).unsqueeze(1).cuda()
    w4 = ((xs - x0) * (ys - y0) * (z1 - zs)).unsqueeze(1).cuda()
    w5 = ((x1 - xs) * (y1 - ys) * (zs - z0)).unsqueeze(1).cuda()
    w6 = ((xs - x0) * (y1 - ys) * (zs - z0)).unsqueeze(1).cuda()
    w7 = ((x1 - xs) * (ys - y0) * (zs - z0)).unsqueeze(1).cuda()
    w8 = ((xs - x0) * (ys - y0) * (zs - z0)).unsqueeze(1).cuda()

    #get values
    if grid_type == 'dense':
        v1 = grid[x0, y0, z0]
        v2 = grid[x1, y0, z0]
        v3 = grid[x0, y1, z0]
        v4 = grid[x1, y1, z0]
        v5 = grid[x0, y0, z1]
        v6 = grid[x1, y0, z1]
        v7 = grid[x0, y1, z1]
        v8 = grid[x1, y1, z1]
    else:
        id1 = (x0 * PRIMES[0] ^ y0 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id2 = (x1 * PRIMES[0] ^ y0 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id3 = (x0 * PRIMES[0] ^ y1 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id4 = (x1 * PRIMES[0] ^ y1 * PRIMES[1] ^ z0 * PRIMES[2]) % grid.shape[0]
        id5 = (x0 * PRIMES[0] ^ y0 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]
        id6 = (x1 * PRIMES[0] ^ y0 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]
        id7 = (x0 * PRIMES[0] ^ y1 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]
        id8 = (x1 * PRIMES[0] ^ y1 * PRIMES[1] ^ z1 * PRIMES[2]) % grid.shape[0]

        v1 = grid[id1]
        v2 = grid[id2]
        v3 = grid[id3]
        v4 = grid[id4]
        v5 = grid[id5]
        v6 = grid[id6]
        v7 = grid[id7]
        v8 = grid[id8]
    #interpolate
    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8
    return out