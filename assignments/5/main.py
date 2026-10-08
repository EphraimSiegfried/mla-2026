import matplotlib.pyplot as plt
import torch
from torchvision.io import ImageReadMode, read_image


def gen_image():
    text = "ML"
    ax = plt.subplot()
    ax.axis("off")
    plt.text(0.5, 0.5, text, size=200, ha="center", va="center")
    plt.savefig(text + ".jpg")
    plt.close()
    return read_image(text + ".jpg", mode=ImageReadMode.GRAY).float().unsqueeze_(0)


def main():
    image = gen_image()
    f1 = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    f2 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    hf = torch.stack([f1, f2])
    hf = hf.unsqueeze(1)

    conv = torch.nn.Conv2d(1, 2, kernel_size=(3, 3), padding=(1, 1), bias=False)
    conv.weight = torch.nn.Parameter(hf, requires_grad=False)
    r = conv(image)
    g_x = r[0, 0]
    g_y = r[0, 1]
    g = torch.sqrt(torch.sum(torch.square(r), dim=1))
    g = g.squeeze()

    plt.imshow(g_x, cmap="gray")
    plt.savefig("figures/sobel_gx.jpg")

    plt.imshow(g_y, cmap="gray")
    plt.savefig("figures/sobel_gy.jpg")

    plt.imshow(g, cmap="gray")
    plt.savefig("figures/sobel_g.jpg")


if __name__ == "__main__":
    main()
