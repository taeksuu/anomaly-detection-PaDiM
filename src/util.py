def concat_embedding_vectors(x, y):
    batch_size, c_x, h_x, w_x = x.size()
    _, c_y, h_y, w_y = y.size()
    s = int(h_x / h_y)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(batch_size, c_x, -1, h_y, w_y)
    z = torch.zeros(batch_size, c_x + c_y, x.size(2), h_y, w_y)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(batch_size, -1, h_y * w_y)
    z = F.fold(z, kernel_size=s, output_size=(h_x, w_x), stride=s)
    
    return z
