import torch
import torchvision

from model.text_encoder import TextEncoder
from model.vae import VAE
from model.unet import UNet
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel
from LoadPretrained import load_vae, load_text, load_unet
from diffusers import DiffusionPipeline
from datasets import load_dataset


# 图像增强模块
compose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(
        512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
    torchvision.transforms.CenterCrop(512),
    # shape = [H, W, C] --> [C, H, W], 像素值 = [0, 255] --> [0.0, 1.0]
    torchvision.transforms.ToTensor(),  # 将PIL.Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    torchvision.transforms.Normalize([0.5], [0.5]),  # 转换为标准正太分布（高斯分布），使模型更容易收敛
])


# 编码数据集
def f(data):
    # 应用图像增强
    pixel_values = [compose(i) for i in data['image']]
    # 文字编码
    input_ids = tokenizer.batch_encode_plus(data['text'],
                                            padding='max_length',
                                            truncation=True,
                                            max_length=77).input_ids
    return {'pixel_values': pixel_values, 'input_ids': input_ids}


# 定义loader
def collate_fn(data):
    pixel_values = [i['pixel_values'] for i in data]
    input_ids = [i['input_ids'] for i in data]
    pixel_values = torch.stack(pixel_values).to(device)
    input_ids = torch.stack(input_ids).to(device)
    return {'pixel_values': pixel_values, 'input_ids': input_ids}


# 计算损失函数
def get_loss(data):
    with torch.no_grad():
        # 文字编码
        # [1, 77] -> [1, 77, 768]
        out_encoder = text_encoder(data['input_ids'])
        # 抽取图像特征图
        # [1, 3, 512, 512] -> [1, 4, 64, 64]
        out_vae = vae.encoder(data['pixel_values'])
        out_vae = vae.sample(out_vae)
        # 0.18215 = vae.config.scaling_factor
        out_vae = out_vae * 0.18215
    # 随机数,unet的计算目标
    noise = torch.randn_like(out_vae)
    # 往特征图中添加噪声
    # 1000 = scheduler.num_train_timesteps
    # 1 = batch size
    noise_step = torch.randint(0, 1000, (1, )).long().to(device)
    out_vae_noise = scheduler.add_noise(out_vae, noise, noise_step)
    # 根据文字信息,把特征图中的噪声计算出来
    out_unet = unet(out_vae=out_vae_noise,
                    out_encoder=out_encoder,
                    time=noise_step)
    # 计算mse loss
    # [1, 4, 64, 64],[1, 4, 64, 64]
    return criterion(out_unet, noise)


def train():
    loss_sum = 0
    for epoch in range(400):
        for i, data in enumerate(loader):
            loss = get_loss(data) / 4
            loss.backward()
            loss_sum += loss.item()
            if (epoch * len(loader) + i) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        if epoch % 10 == 0:
            print(epoch, loss_sum)
            loss_sum = 0
    torch.save(unet.to('cpu'), 'model/unet.model')  # 整个模型，大约3.2G


if __name__ == '__main__':
    # 模型初始化
    vae = VAE()
    text_encoder = TextEncoder()
    unet = UNet()
    print('模型初始化成功')

    # 加载预训练模型参数
    params_vae = AutoencoderKL.from_pretrained(
        'lansinuote/diffsion_from_scratch.params', subfolder='vae')
    params_text = CLIPTextModel.from_pretrained(
        'lansinuote/diffsion_from_scratch.params', subfolder='text_encoder')
    params_unet = UNet2DConditionModel.from_pretrained(
        'lansinuote/diffsion_from_scratch.params', subfolder='unet')
    print('加载预训练模型参数成功')

    # 将预训练参数读入模型
    load_vae(vae, params_vae)
    load_text(text_encoder, params_text)
    load_unet(unet, params_unet)
    print('预训练参数读入模型成功')

    # 只训练unet，冻结text encoder和vae
    vae.eval()  # 测试模式：数据不进行反向传播，但仍需计算梯度
    text_encoder.eval()
    unet.train()  # 训练模式
    vae.requires_grad_(False)  # 不计算梯度
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    #  模型放入指定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # 设置优化器
    optimizer = torch.optim.AdamW(unet.parameters(),
                                  lr=1e-5,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01,
                                  eps=1e-8)
    criterion = torch.nn.MSELoss()

    # 加载工具类
    pipeline = DiffusionPipeline.from_pretrained(
        'lansinuote/diffsion_from_scratch.params', safety_checker=None)
    scheduler = pipeline.scheduler  # 噪声添加器
    tokenizer = pipeline.tokenizer  # 分词器
    del pipeline
    print('工具类加载成功')

    # 加载数据集
    dataset = load_dataset(path='lansinuote/diffsion_from_scratch', split='train')
    dataset = dataset.map(f,
                          batched=True,
                          batch_size=100,
                          num_proc=1,
                          remove_columns=['image', 'text'])  # map:对数据集中每个数据进行f编码，一次编码100个
    dataset.set_format(type='torch')  # 转为PyTorch数据集格式
    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=True,
                                         collate_fn=collate_fn,
                                         batch_size=1)
    print('数据集加载成功')

    # 开始训练
    train()
