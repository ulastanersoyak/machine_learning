import torch
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def train_one_epoch(model,train_dataloader,loss_fn,optimizer,device):
    model.to(device)
    model.train()
    for images in train_dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, images)
        loss.backward()
        optimizer.step()
    print(f"loss -> {loss}")

def train(model,train_dataloader,loss_fn,optimizer,device,epochs):
    for i in range(epochs):
        print(f"epoch-> {i+1}")
        train_one_epoch(model,train_dataloader,loss_fn,optimizer,device)
        print("------------------------------------------")
    print("TRAINING DONE!")
    torch.save(model.state_dict(), "autoencoder.pth")

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    model.to(device)
    total_loss = 0
    total_ssim = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            total_loss += loss.item()
            for j in range(inputs.shape[0]):
                img1 = inputs[j].cpu().numpy().transpose((1, 2, 0))
                img2 = outputs[j].cpu().numpy().transpose((1, 2, 0))
                total_ssim += ssim(img1, img2, multichannel=True)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_ssim = total_ssim / (len(dataloader.dataset) * inputs.shape[-1] * inputs.shape[-2])
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    return avg_loss, avg_ssim

def test_with_input(model,img_path):
    img = Image.open(img_path).resize((256,256))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    encoded = model.encoder(img_tensor)
    reconstructed = model.decoder(encoded)
    reconstructed = reconstructed.squeeze(0)
    reconstructed_img = transforms.ToPILImage()(reconstructed)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[1].imshow(reconstructed_img)
    axs[1].set_title("Reconstructed")
    plt.show()