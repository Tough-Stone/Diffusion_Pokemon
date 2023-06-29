import torch


# res->vae
def load_res_vae(model, param):
    model.s[0].load_state_dict(param.norm1.state_dict())
    model.s[2].load_state_dict(param.conv1.state_dict())
    model.s[3].load_state_dict(param.norm2.state_dict())
    model.s[5].load_state_dict(param.conv2.state_dict())

    if isinstance(model.res, torch.nn.Module):
        model.res.load_state_dict(param.conv_shortcut.state_dict())


# attention->vae
def load_atten(model, param):
    model.norm.load_state_dict(param.group_norm.state_dict())
    model.q.load_state_dict(param.query.state_dict())
    model.k.load_state_dict(param.key.state_dict())
    model.v.load_state_dict(param.value.state_dict())
    model.out.load_state_dict(param.proj_attn.state_dict())


# vae:encoder+decoder
def load_vae(model, params):
    # encoder.in
    model.encoder[0].load_state_dict(params.encoder.conv_in.state_dict())
    # encoder.down
    for i in range(4):
        load_res_vae(model.encoder[i + 1][0], params.encoder.down_blocks[i].resnets[0])
        load_res_vae(model.encoder[i + 1][1], params.encoder.down_blocks[i].resnets[1])
        if i != 3:
            model.encoder[i + 1][2][1].load_state_dict(
                params.encoder.down_blocks[i].downsamplers[0].conv.state_dict())
    # encoder.mid
    load_res_vae(model.encoder[5][0], params.encoder.mid_block.resnets[0])
    load_res_vae(model.encoder[5][2], params.encoder.mid_block.resnets[1])
    load_atten(model.encoder[5][1], params.encoder.mid_block.attentions[0])
    # encoder.out
    model.encoder[6][0].load_state_dict(params.encoder.conv_norm_out.state_dict())
    model.encoder[6][2].load_state_dict(params.encoder.conv_out.state_dict())
    # encoder.正态分布层
    model.encoder[7].load_state_dict(params.quant_conv.state_dict())
    # decoder.正态分布层
    model.decoder[0].load_state_dict(params.post_quant_conv.state_dict())
    # decoder.in
    model.decoder[1].load_state_dict(params.decoder.conv_in.state_dict())
    # decoder.mid
    load_res_vae(model.decoder[2][0], params.decoder.mid_block.resnets[0])
    load_res_vae(model.decoder[2][2], params.decoder.mid_block.resnets[1])
    load_atten(model.decoder[2][1], params.decoder.mid_block.attentions[0])
    # decoder.up
    for i in range(4):
        load_res_vae(model.decoder[i + 3][0], params.decoder.up_blocks[i].resnets[0])
        load_res_vae(model.decoder[i + 3][1], params.decoder.up_blocks[i].resnets[1])
        load_res_vae(model.decoder[i + 3][2], params.decoder.up_blocks[i].resnets[2])
        if i != 3:
            model.decoder[i + 3][4].load_state_dict(
                params.decoder.up_blocks[i].upsamplers[0].conv.state_dict())
    # decoder.out
    model.decoder[7][0].load_state_dict(params.decoder.conv_norm_out.state_dict())
    model.decoder[7][2].load_state_dict(params.decoder.conv_out.state_dict())


# text encoder:词嵌入+12×clip encoder
def load_text(model, params):
    model.embed.embed.load_state_dict(
        params.text_model.embeddings.token_embedding.state_dict())
    model.embed.pos_embed.load_state_dict(
        params.text_model.embeddings.position_embedding.state_dict())
    for i in range(12):
        # 第一层norm
        model.encoder[i].s1[0].load_state_dict(
            params.text_model.encoder.layers[i].layer_norm1.state_dict())
        # 注意力q矩阵
        model.encoder[i].s1[1].q.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.q_proj.state_dict())
        # 注意力k矩阵
        model.encoder[i].s1[1].k.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.k_proj.state_dict())
        # 注意力v矩阵
        model.encoder[i].s1[1].v.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.v_proj.state_dict())
        # 注意力out
        model.encoder[i].s1[1].out.load_state_dict(
            params.text_model.encoder.layers[i].self_attn.out_proj.state_dict())
        # 第二层norm
        model.encoder[i].s2[0].load_state_dict(
            params.text_model.encoder.layers[i].layer_norm2.state_dict())
        # mlp第一层fc
        model.encoder[i].s2[1].load_state_dict(
            params.text_model.encoder.layers[i].mlp.fc1.state_dict())
        # mlp第二层fc
        model.encoder[i].s3.load_state_dict(
            params.text_model.encoder.layers[i].mlp.fc2.state_dict())
    model.encoder[12].load_state_dict(params.text_model.final_layer_norm.state_dict())


# transformer->down block / up block->unet
def load_tf(model, param):
    model.norm_in.load_state_dict(param.norm.state_dict())
    model.cnn_in.load_state_dict(param.proj_in.state_dict())
    model.atten1.q.load_state_dict(
        param.transformer_blocks[0].attn1.to_q.state_dict())
    model.atten1.k.load_state_dict(
        param.transformer_blocks[0].attn1.to_k.state_dict())
    model.atten1.v.load_state_dict(
        param.transformer_blocks[0].attn1.to_v.state_dict())
    model.atten1.out.load_state_dict(
        param.transformer_blocks[0].attn1.to_out[0].state_dict())
    model.atten2.q.load_state_dict(
        param.transformer_blocks[0].attn2.to_q.state_dict())
    model.atten2.k.load_state_dict(
        param.transformer_blocks[0].attn2.to_k.state_dict())
    model.atten2.v.load_state_dict(
        param.transformer_blocks[0].attn2.to_v.state_dict())
    model.atten2.out.load_state_dict(
        param.transformer_blocks[0].attn2.to_out[0].state_dict())
    model.fc0.load_state_dict(
        param.transformer_blocks[0].ff.net[0].proj.state_dict())
    model.fc1.load_state_dict(
        param.transformer_blocks[0].ff.net[2].state_dict())
    model.norm_atten0.load_state_dict(
        param.transformer_blocks[0].norm1.state_dict())
    model.norm_atten1.load_state_dict(
        param.transformer_blocks[0].norm2.state_dict())
    model.norm_act.load_state_dict(
        param.transformer_blocks[0].norm3.state_dict())

    model.cnn_out.load_state_dict(param.proj_out.state_dict())


# resnet->down block / up block->unet
def load_res_unet(model, param):
    model.time[1].load_state_dict(param.time_emb_proj.state_dict())
    model.s0[0].load_state_dict(param.norm1.state_dict())
    model.s0[2].load_state_dict(param.conv1.state_dict())
    model.s1[0].load_state_dict(param.norm2.state_dict())
    model.s1[2].load_state_dict(param.conv2.state_dict())
    if isinstance(model.res, torch.nn.Module):
        model.res.load_state_dict(param.conv_shortcut.state_dict())


# down block->unet
def load_down_block(model, param):
    load_tf(model.tf0, param.attentions[0])
    load_tf(model.tf1, param.attentions[1])
    load_res_unet(model.res0, param.resnets[0])
    load_res_unet(model.res1, param.resnets[1])
    model.out.load_state_dict(param.downsamplers[0].conv.state_dict())


# up block->unet
def load_up_block(model, param):
    load_tf(model.tf0, param.attentions[0])
    load_tf(model.tf1, param.attentions[1])
    load_tf(model.tf2, param.attentions[2])
    load_res_unet(model.res0, param.resnets[0])
    load_res_unet(model.res1, param.resnets[1])
    load_res_unet(model.res2, param.resnets[2])
    if isinstance(model.out, torch.nn.Module):
        model.out[1].load_state_dict(param.upsamplers[0].conv.state_dict())


# unet模型：1层in, 4层down, 1层middle, 4层up, 1层out
def load_unet(model, param):
    # in
    model.in_vae.load_state_dict(param.conv_in.state_dict())
    model.in_time[0].load_state_dict(param.time_embedding.linear_1.state_dict())
    model.in_time[2].load_state_dict(param.time_embedding.linear_2.state_dict())
    # down
    load_down_block(model.down_block0, param.down_blocks[0])
    load_down_block(model.down_block1, param.down_blocks[1])
    load_down_block(model.down_block2, param.down_blocks[2])
    load_res_unet(model.down_res0, param.down_blocks[3].resnets[0])
    load_res_unet(model.down_res1, param.down_blocks[3].resnets[1])
    # mid
    load_tf(model.mid_tf, param.mid_block.attentions[0])
    load_res_unet(model.mid_res0, param.mid_block.resnets[0])
    load_res_unet(model.mid_res1, param.mid_block.resnets[1])
    # up
    load_res_unet(model.up_res0, param.up_blocks[0].resnets[0])
    load_res_unet(model.up_res1, param.up_blocks[0].resnets[1])
    load_res_unet(model.up_res2, param.up_blocks[0].resnets[2])
    model.up_in[1].load_state_dict(
        param.up_blocks[0].upsamplers[0].conv.state_dict())
    load_up_block(model.up_block0, param.up_blocks[1])
    load_up_block(model.up_block1, param.up_blocks[2])
    load_up_block(model.up_block2, param.up_blocks[3])
    # out
    model.out[0].load_state_dict(param.conv_norm_out.state_dict())
    model.out[2].load_state_dict(param.conv_out.state_dict())