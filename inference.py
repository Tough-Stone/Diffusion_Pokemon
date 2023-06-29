import torch
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
from model.vae import VAE
from model.text_encoder import TextEncoder
from model.unet import UNet
from transformers import PreTrainedModel, PretrainedConfig
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel
from LoadPretrained import load_vae, load_text, load_unet


# 包装类
class Model(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.unet = unet.to('cpu')


@torch.no_grad()
def generate(text):
    # 词编码
    # [1, 77]
    pos = tokenizer(text,
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)
    neg = tokenizer('',
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt').input_ids.to(device)

    # [1, 77, 768]
    pos = text_encoder(pos)
    neg = text_encoder(neg)
    print('完成词编码')
    # [1+1, 77, 768] -> [2, 77, 768]
    out_encoder = torch.cat((neg, pos), dim=0)

    # vae的压缩图,从随机噪声开始
    out_vae = torch.randn(1, 4, 64, 64, device=device)
    print('生成随机噪声')
    # 生成50个时间步,一般是从980-0
    scheduler.set_timesteps(20, device=device)
    for time in scheduler.timesteps:
        # 往图中加噪音
        # [1+1, 4, 64, 64] -> [2, 4, 64, 64]
        noise = torch.cat((out_vae, out_vae), dim=0)
        noise = scheduler.scale_model_input(noise, time)

        # 计算噪音
        # [2, 4, 64, 64],[2, 77, 768],scala -> [2, 4, 64, 64]
        pred_noise = unet(out_vae=noise, out_encoder=out_encoder, time=time)

        # 从正例图中减去反例图
        # [2, 4, 64, 64] -> [1, 4, 64, 64]
        pred_noise = pred_noise[0] + 7.5 * (pred_noise[1] - pred_noise[0])

        # 重新添加噪音,以进行下一步计算
        # [1, 4, 64, 64]
        out_vae = scheduler.step(pred_noise, time, out_vae).prev_sample

    # 从压缩图恢复成图片
    out_vae = 1 / 0.18215 * out_vae
    # [1, 4, 64, 64] -> [1, 3, 512, 512]
    image = vae.decoder(out_vae)

    # 转换成图片数据
    image = image.cpu()
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(0, 2, 3, 1)
    print('图像格式转换完成')
    return image.numpy()[0]


# 画图
def show():
    # texts = [
    #     'a drawing of a star with a jewel in the center',  # 宝石海星
    #     'a drawing of a woman in a red cape',  # 迷唇姐
    #     'a drawing of a dragon sitting on its hind legs',  # 肥大
    #     'a drawing of a blue sea turtle holding a rock',  # 拉普拉斯
    #     'a blue and white bird with its wings spread',  # 急冻鸟
    #     'a blue and white stuffed animal sitting on top of a white surface',  # 卡比兽
    # ]
    text = 'A man in red with a sword'

    # images = [generate(i) for i in texts]
    image = generate(text)

    plt.figure(figsize=(10, 5))
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.imshow(images[i])
    #     plt.axis('off')

    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 模型初始化
    vae = VAE()
    text_encoder = TextEncoder()
    unet = UNet()
    vae.eval()
    text_encoder.eval()
    unet.eval()
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

    # 加载工具类
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = DiffusionPipeline.from_pretrained(
        'lansinuote/diffsion_from_scratch.params', safety_checker=None)
    scheduler = pipeline.scheduler
    tokenizer = pipeline.tokenizer
    del pipeline
    print('工具类加载成功')

    show()

    # 加载unet预训练模型
    unet = Model.from_pretrained('lansinuote/diffsion_from_scratch.unet').unet
    unet.eval().to(device)
    print('pokeman预训练模型加载成功')
    show()

