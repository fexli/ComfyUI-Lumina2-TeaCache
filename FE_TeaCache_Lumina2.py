import torch
import numpy as np
from comfy.ldm.common_dit import pad_to_patch_size  # noqa

from unittest.mock import patch


# referenced from https://github.com/spawner1145/TeaCache/blob/main/TeaCache4Lumina2/teacache_lumina2.py
# transplanted by @fexli
def teacache_forward_working(
        self, x, timesteps, context, num_tokens, attention_mask=None, transformer_options={}, **kwargs
):
    if not hasattr(self, 'teacache_state'):
        self.teacache_state = {
            "cnt": 0,
            "num_steps": transformer_options.get("num_steps"),
            "cache": transformer_options.get("cache"),
            "uncond_seq_len": transformer_options.get("uncond_seq_len")
        }
    # 初始化TeaCache相关参数
    cap_feats = context
    cap_mask = attention_mask
    bs, c, h, w = x.shape
    x = pad_to_patch_size(x, (self.patch_size, self.patch_size))
    t = (1.0 - timesteps).to(dtype=x.dtype)

    # 时间嵌入处理
    t_emb = self.t_embedder(t, dtype=x.dtype)
    adaln_input = t_emb

    # 文本特征嵌入
    cap_feats = self.cap_embedder(cap_feats)

    # 图像分块嵌入和位置编码
    x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t_emb, num_tokens)
    freqs_cis = freqs_cis.to(x.device)

    # TeaCache核心逻辑
    max_seq_len = x.shape[1]
    should_calc = True

    enable_teacache = transformer_options.get('transformer_options', True)

    if enable_teacache:
        # 初始化缓存
        cache_key = max_seq_len
        if cache_key not in self.teacache_state['cache']:
            self.teacache_state['cache'][cache_key] = {
                "accumulated_rel_l1_distance": 0.0,
                "previous_modulated_input": None,
                "previous_residual": None,
            }
        current_cache = self.teacache_state['cache'][cache_key]

        # 计算调制输入
        modulated_inp = self.layers[0].adaLN_modulation(adaln_input.clone())[0]

        # 缓存更新逻辑
        if self.teacache_state['cnt'] == 0 or self.teacache_state['cnt'] == self.teacache_state['num_steps'] - 1:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            if current_cache["previous_modulated_input"] is not None:
                # 多项式系数调整
                coefficients = [393.76566581, -603.50993606, 209.10239044, -23.00726601,
                                0.86377344]  # taken from teacache_lumina_next.py
                rescale_func = np.poly1d(coefficients)

                # 计算相对L1变化
                prev_mod_input = current_cache["previous_modulated_input"]
                prev_mean = prev_mod_input.abs().mean()
                if prev_mean.item() > 1e-9:
                    rel_l1_change = ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).cpu().item()
                else:
                    rel_l1_change = 0.0 if modulated_inp.abs().mean().item() < 1e-9 else float('inf')

                # 累计变化量
                current_cache["accumulated_rel_l1_distance"] += rescale_func(rel_l1_change)

                # 阈值判断
                if current_cache["accumulated_rel_l1_distance"] < transformer_options['rel_l1_thresh']:
                    should_calc = False
                else:
                    should_calc = True
                    current_cache["accumulated_rel_l1_distance"] = 0.0

        current_cache["previous_modulated_input"] = modulated_inp.clone()

        # 序列长度管理
        if self.teacache_state['uncond_seq_len'] is None:
            self.teacache_state['uncond_seq_len'] = cache_key
        if cache_key != self.teacache_state['uncond_seq_len']:
            self.teacache_state['cnt'] += 1
            if self.teacache_state['cnt'] >= self.teacache_state['num_steps']:
                self.teacache_state['cnt'] = 0

    # 主处理流程
    if enable_teacache and not should_calc:
        processed_x = x + current_cache["previous_residual"]
    else:
        original_x = x.clone()
        current_x = x
        for layer in self.layers:
            current_x = layer(current_x, mask, freqs_cis, adaln_input)

        if enable_teacache:
            current_cache["previous_residual"] = current_x - original_x
        processed_x = current_x

    if enable_teacache:
        del current_cache
    # 最终输出处理
    output = self.final_layer(processed_x, adaln_input)
    output = self.unpatchify(output, img_size, cap_size, return_tensor=True)[:, :, :h, :w]

    return -output


class FE_TeaCache_Lumina2:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.001}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "The start percentage of the steps that will apply TeaCache."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "The end percentage of the steps that will apply TeaCache."})

            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_teacache"
    CATEGORY = "fexli/utils"
    EXPERIMENTAL = True

    def patch_teacache(self, model, rel_l1_thresh, steps, start_percent, end_percent):
        # start_percent = 0.0
        # end_percent = 1.0

        if rel_l1_thresh == 0:
            return (model,)

        # 克隆模型并获取transformer
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}

        # 注入TeaCache属性
        new_model.model_options["transformer_options"]["enable_teacache"] = True
        new_model.model_options["transformer_options"]["cnt"] = 0
        new_model.model_options["transformer_options"]["num_steps"] = steps
        new_model.model_options["transformer_options"]["cache"] = {}
        new_model.model_options["transformer_options"]["uncond_seq_len"] = None
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        diffusion_model = new_model.get_model_object("diffusion_model")

        context = patch.multiple(
            diffusion_model,
            forward=teacache_forward_working.__get__(diffusion_model, diffusion_model.__class__)
        )

        # referenced from https://github.com/welltop-cn/ComfyUI-TeaCache/blob/main/nodes.py
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs["cond_or_uncond"]
            # referenced from https://github.com/kijai/ComfyUI-KJNodes/blob/d126b62cebee81ea14ec06ea7cd7526999cb0554/nodes/model_optimization_nodes.py#L868
            sigmas = c["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break

            if current_step_index == 0:
                if (1 in cond_or_uncond) and hasattr(diffusion_model, 'teacache_state'):
                    delattr(diffusion_model, 'teacache_state')

            current_percent = current_step_index / (len(sigmas) - 1)
            if start_percent <= current_percent <= end_percent:
                c["transformer_options"]["enable_teacache"] = True
            else:
                c["transformer_options"]["enable_teacache"] = False

            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)
