import os
import gc
import shutil
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
try:
    import spaces
except ImportError:
    # `spaces` only supplies the ZeroGPU decorator on Hugging Face Spaces.
    class _LocalSpaces:
        @staticmethod
        def GPU(*_args, **_kwargs):
            return lambda fn: fn

    spaces = _LocalSpaces()
import torch
import random
import base64
import json
import secrets
from io import BytesIO
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.version.hip =", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16

if device.type != "cuda":
    raise RuntimeError("A CUDA or ROCm GPU is required; PyTorch did not expose one as `cuda`.")

is_rocm = torch.version.hip is not None
use_cpu_offload = os.environ.get("QWEN_CPU_OFFLOAD", "1" if is_rocm else "0") == "1"

# Rapid-AIO V19 is stored as FP8. Loading it with torch_dtype=bfloat16
# silently expands it from ~20 GiB to ~40 GiB and can kill a 32+32 GiB
# Strix Halo machine while the checkpoint shards are being materialized.
transformer = QwenImageTransformer2DModel.from_pretrained(
    "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
    torch_dtype=torch.float8_e4m3fn,
    low_cpu_mem_usage=True,
    device_map="cuda",
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=dtype,
)

# enable_layerwise_casting first converts the whole model to storage_dtype and
# then skips installing conversion hooks on normalization/embedding/output
# modules. Rapid-AIO stores those leaves as FP8 too, so upcast them *after* the
# hook setup; otherwise txt_norm tries Float × Float8 on ROCm.
precision_patterns = ("norm", "pos_embed", "patch_embed")
precision_exact = {"proj_in", "proj_out"}
upcast_tensors = 0
for module_name, module in transformer.named_modules():
    leaf_name = module_name.rsplit(".", 1)[-1]
    if any(pattern in module_name for pattern in precision_patterns) or leaf_name in precision_exact:
        for parameter in module.parameters(recurse=False):
            if parameter.is_floating_point() and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)
                upcast_tensors += 1
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point() and buffer.dtype != dtype:
                setattr(module, buffer_name, buffer.to(dtype=dtype))
                upcast_tensors += 1
print(f"Upcast {upcast_tensors} precision-sensitive transformer tensors to {dtype}.")

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=transformer,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

if use_cpu_offload:
    offload_dir = os.environ.get(
        "QWEN_OFFLOAD_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".offload-cache"),
    )
    # Diffusers creates uniquely named files and does not clean old runs. A
    # stale cache wastes many GiB and can preserve tensors from a bad launch.
    shutil.rmtree(offload_dir, ignore_errors=True)
    os.makedirs(offload_dir, exist_ok=True)
    print(f"Enabling leaf-level disk offload: {offload_dir}")
    pipe.enable_group_offload(
        onload_device=device,
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        offload_to_disk_path=offload_dir,
        use_stream=False,
    )
else:
    pipe.to(device)

from diffusers.hooks.group_offloading import GroupOffloadingHook

# Keep the original, valid leaf-offload groups for the lifetime of the
# process. PEFT replaces Linear modules in-place; taking a fresh snapshot
# after one adapter is installed captures wrapper/base_layer names instead of
# this canonical module layout and corrupts subsequent adapter switches.
BASE_OFFLOAD_GROUPS = {}
if use_cpu_offload:
    for base_name, base_module in pipe.transformer.named_modules():
        registry = getattr(base_module, "_diffusers_hook", None)
        hook = registry.get_hook("group_offloading") if registry is not None else None
        if hook is not None:
            BASE_OFFLOAD_GROUPS[base_name] = (hook.group, hook.config, hook.next_group)
    print(f"Saved {len(BASE_OFFLOAD_GROUPS)} canonical base-model offload groups.")


def restore_base_offload_groups() -> int:
    """Discard PEFT/reapply copies and restore canonical base-model groups."""
    if not use_cpu_offload:
        return 0

    # unload_lora() and PEFT injection can both reapply group offloading. Drop
    # every generated copy before restoring the original groups and caches.
    for _, module in pipe.transformer.named_modules():
        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            continue
        for hook_name in ("layer_execution_tracker", "lazy_prefetch_group_offloading", "group_offloading"):
            if registry.get_hook(hook_name) is not None:
                registry.remove_hook(hook_name, recurse=False)

    restored = 0
    for original_name, (group, config, next_group) in BASE_OFFLOAD_GROUPS.items():
        module = pipe.transformer.get_submodule(original_name)
        # PEFT wraps the original Linear. Offload the actual base layer, never
        # the wrapper or its adapter leaves.
        module = getattr(module, "base_layer", module)
        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            continue
        restored_hook = GroupOffloadingHook(group, config=config)
        restored_hook.next_group = next_group
        registry.register_hook(restored_hook, "group_offloading")
        restored += 1
    return restored

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

ADAPTER_SPECS = {
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles",
    },
    "Photo-to-Anime": {
        "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        "adapter_name": "photo-to-anime",
    },
    "Anime-V2": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "weights": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
        "adapter_name": "anime-v2",
    },
    "Light-Migration": {
        "repo": "dx8152/Qwen-Edit-2509-Light-Migration",
        "weights": "参考色调.safetensors",
        "adapter_name": "light-migration",
    },
    "Upscaler": {
        "repo": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        "weights": "qwen_image_edit_2511_upscale.safetensors",
        "adapter_name": "upscale-2k",
    },
    "Style-Transfer": {
        "repo": "zooeyy/Style-Transfer",
        "weights": "Style Transfer-Alpha-V0.1.safetensors",
        "adapter_name": "style-transfer",
    },
    "Manga-Tone": {
        "repo": "nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
        "weights": "tone001.safetensors",
        "adapter_name": "manga-tone",
    },
    "Anything2Real": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601.safetensors",
        "adapter_name": "anything2real",
    },
    "Fal-Multiple-Angles": {
        "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "weights": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "adapter_name": "fal-multiple-angles",
    },
    "Polaroid-Photo": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
        "weights": "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
        "adapter_name": "polaroid-photo",
    },
    "Unblur-Anything": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
        "weights": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
        "adapter_name": "unblur-anything",
    },
    "Midnight-Noir-Eyes-Spotlight": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
        "weights": "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
        "adapter_name": "midnight-noir-eyes-spotlight",
    },
    "Hyper-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
        "adapter_name": "hyper-realistic-portrait",
    },
    "Ultra-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
        "adapter_name": "ultra-realistic-portrait",
    },
    "Pixar-Inspired-3D": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
        "weights": "PI3_20.safetensors",
        "adapter_name": "pi3",
    },
    "Noir-Comic-Book": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
        "weights": "Noir-Comic-Book-Panel_20.safetensors",
        "adapter_name": "ncb",
    },
    "Any-light": {
        "repo": "lilylilith/QIE-2511-MP-AnyLight",
        "weights": "QIE-2511-AnyLight_.safetensors",
        "adapter_name": "any-light",
    },
    "Studio-DeLight": {
        "repo": "prithivMLmods/QIE-2511-Studio-DeLight",
        "weights": "QIE-2511-Studio-DeLight-5000.safetensors",
        "adapter_name": "studio-delight",
    },
    "Cinematic-FlatLog": {
        "repo": "prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
        "weights": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
        "adapter_name": "flat-log",
    },
}

LOADED_ADAPTERS: set = set()
ADAPTER_NAMES = list(ADAPTER_SPECS.keys())

EXAMPLES_CONFIG = [
    {"images": ["examples/B.jpg"],                          "prompt": "Transform into anime.",                                                                                           "lora": "Photo-to-Anime"},
    {"images": ["examples/HRP.jpg"],                        "prompt": "Transform into a hyper-realistic face portrait.",                                                                 "lora": "Hyper-Realistic-Portrait"},
    {"images": ["examples/A.jpeg"],                         "prompt": "Rotate the camera 45 degrees to the right.",                                                                      "lora": "Multiple-Angles"},
    {"images": ["examples/U.jpg"],                          "prompt": "Upscale this picture to 4K resolution.",                                                                          "lora": "Upscaler"},
    {"images": ["examples/L1.jpg", "examples/L2.jpg"],      "prompt": "Apply the lighting from image 2 to image 1.",                                                                     "lora": "Any-light"},
    {"images": ["examples/PP1.jpg"],                        "prompt": "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed preserving realistic texture and details.", "lora": "Polaroid-Photo"},
    {"images": ["examples/Z1.jpg"],                         "prompt": "Front-right quarter view.",                                                                                       "lora": "Fal-Multiple-Angles"},
    {"images": ["examples/URP.jpg"],                        "prompt": "Transform into a cinematic flat log.",                                                                            "lora": "Cinematic-FlatLog"},
    {"images": ["examples/SL.jpg"],                         "prompt": "Neutral uniform lighting. Preserve identity and composition.",                                                    "lora": "Studio-DeLight"},
    {"images": ["examples/PI.jpg"],                         "prompt": "Transform it into Pixar-inspired 3D.",                                                                            "lora": "Pixar-Inspired-3D"},
    {"images": ["examples/MT.jpg"],                         "prompt": "Paint with manga tone.",                                                                                          "lora": "Manga-Tone"},
    {"images": ["examples/NCB.jpg"],                        "prompt": "Transform into a noir comic book style.",                                                                         "lora": "Noir-Comic-Book"},
    {"images": ["examples/URP.jpg"],                        "prompt": "Ultra-realistic portrait.",                                                                                       "lora": "Ultra-Realistic-Portrait"},
    {"images": ["examples/MN.jpg"],                         "prompt": "Transform into Midnight Noir Eyes Spotlight.",                                                                    "lora": "Midnight-Noir-Eyes-Spotlight"},
    {"images": ["examples/ST1.jpg", "examples/ST2.jpg"],    "prompt": "Convert Image 1 to the style of Image 2.",                                                                        "lora": "Style-Transfer"},
    {"images": ["examples/R1.jpg"],                         "prompt": "Change the picture to realistic photograph.",                                                                     "lora": "Anything2Real"},
    {"images": ["examples/UA.jpeg"],                        "prompt": "Unblur and upscale.",                                                                                             "lora": "Unblur-Anything"},
    {"images": ["examples/L1.jpg", "examples/L2.jpg"],      "prompt": "Refer to the color tone, remove the original lighting from Image 1, and relight Image 1 based on the lighting and color tone of Image 2.", "lora": "Light-Migration"},
    {"images": ["examples/P1.jpg"],                         "prompt": "Transform into anime (while preserving the background and remaining elements maintaining realism and original details.)", "lora": "Anime-V2"},
]


def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""


def build_client_config():
    """Static config consumed by the frontend: LoRA list + example cards."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumbs": [make_thumb_b64(p) for p in ex["images"]],
            "n_images": len(ex["images"]),
            "lora": ex["lora"],
            "prompt": ex["prompt"],
        })
    return {
        "loras": ADAPTER_NAMES,
        "default_lora": "Photo-to-Anime",
        "examples": examples,
    }


print("Building client config (example thumbnails)…")
CLIENT_CONFIG = build_client_config()
print(f"Built config with {len(EXAMPLES_CONFIG)} examples and {len(ADAPTER_NAMES)} LoRAs.")


def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images


def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    w, h = image.size
    if w > h:
        nw = 1024
        nh = int(nw * h / w)
    else:
        nh = 1024
        nw = int(nh * w / h)
    return (nw // 8) * 8, (nh // 8) * 8


def peft_parameter_name(source_name: str, adapter_name: str) -> str:
    """Map common LoRA safetensors key layouts to PEFT parameter names."""
    target_name = source_name.removeprefix("diffusion_model.")
    for projection in ("lora_A", "lora_B"):
        target_name = target_name.replace(
            f".{projection}.default.weight",
            f".{projection}.{adapter_name}.weight",
        )
        target_name = target_name.replace(
            f".{projection}.weight",
            f".{projection}.{adapter_name}.weight",
        )
    return target_name


# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="Qwen-Image-Edit-2511-LoRAs-Fast")


@app.middleware("http")
async def require_basic_auth(request, call_next):
    """Protect custom FastAPI routes as well as Gradio endpoints on the LAN."""
    expected_user = os.environ.get("QWEN_AUTH_USER")
    expected_password = os.environ.get("QWEN_AUTH_PASSWORD")
    if not expected_user or not expected_password:
        return await call_next(request)
    # Gradio performs its own startup probes over localhost. They must bypass
    # LAN authentication or launch() aborts before the server becomes ready.
    if request.client and request.client.host in ("127.0.0.1", "::1"):
        return await call_next(request)

    authorization = request.headers.get("authorization", "")
    valid = False
    if authorization.startswith("Basic "):
        try:
            decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
            valid = secrets.compare_digest(username, expected_user) and secrets.compare_digest(
                password, expected_password
            )
        except (ValueError, UnicodeDecodeError):
            pass

    if not valid:
        return HTMLResponse(
            "Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Qwen Image Edit"'},
        )
    return await call_next(request)


@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    images_b64_json: str,
    prompt: str,
    lora_adapter: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    steps: int,
) -> dict:
    """Edit one or more images with Qwen-Image-Edit-2511 + a lazily-loaded LoRA.

    Returns {"image": <base64 PNG data URL>, "seed": <seed used>}.
    """
    gc.collect()
    torch.cuda.empty_cache()

    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image to edit.")
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter an edit prompt.")

    spec = ADAPTER_SPECS.get(lora_adapter)
    if not spec:
        raise gr.Error(f"Configuration not found for: {lora_adapter}")

    adapter_name = spec["adapter_name"]
    newly_loaded_adapter = False
    if adapter_name not in LOADED_ADAPTERS:
        print(f"--- Downloading and Loading Adapter: {lora_adapter} ---")
        try:
            # Multiple simultaneously injected adapters cause PEFT wrappers to
            # nest around an already-offloaded model. Keep exactly one adapter
            # resident and return to the canonical base layout before loading
            # a different one.
            if LOADED_ADAPTERS:
                previous = ", ".join(sorted(LOADED_ADAPTERS))
                print(f"Unloading active adapter(s) before switch: {previous}")
                pipe.unload_lora_weights()
                LOADED_ADAPTERS.clear()
                restored = restore_base_offload_groups()
                print(f"Restored {restored} canonical offload groups after adapter unload.")

            pipe.load_lora_weights(spec["repo"], weight_name=spec["weights"], adapter_name=adapter_name)
            restored_base_hooks = restore_base_offload_groups()
            print(f"Restored {restored_base_hooks} original base-layer offload hooks after PEFT injection.")
            LOADED_ADAPTERS.add(adapter_name)
            newly_loaded_adapter = True
        except Exception as e:
            raise gr.Error(f"Failed to load adapter {lora_adapter}: {e}")
    else:
        print(f"--- Adapter {lora_adapter} already loaded. ---")

    pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    # PEFT infers a newly-created LoRA's dtype from its FP8 base layer, and
    # set_adapters() can move/recast it again. Diffusers also assigns the new
    # LoRA leaves an FP8-storage layerwise hook. Pin that hook and its tensors
    # only after activation so ROCm receives matching BF16 inputs and weights.
    fixed_lora_hooks = 0
    detached_offload_hooks = 0
    detached_wrapper_hooks = 0
    fp32_modulation_hooks = 0
    for module_name, module in pipe.transformer.named_modules():
        # PEFT replaces each Linear with a wrapper and copies the original
        # Diffusers hook registry onto that wrapper while retaining the real
        # hooked Linear as base_layer. With disk offload this creates a second
        # cache from offloaded placeholders (zero weights and NaN FP8 biases).
        # Keep only base_layer's original offload hook.
        if hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "_diffusers_hook"):
            for hook_name in ("layer_execution_tracker", "lazy_prefetch_group_offloading", "group_offloading"):
                if module._diffusers_hook.get_hook(hook_name) is not None:
                    module._diffusers_hook.remove_hook(hook_name, recurse=False)
                    detached_wrapper_hooks += 1

        if module_name.endswith(("img_mod.1.base_layer", "txt_mod.1.base_layer")) and hasattr(
            module, "_diffusers_hook"
        ):
            casting_hook = module._diffusers_hook.get_hook("layerwise_casting")
            if casting_hook is not None:
                # gfx1151 rocBLAS occasionally emits isolated NaNs for this
                # M=1,N=18432,K=3072 BF16 GEMM. Keep FP8 storage, but compute
                # the modulation projections in FP32.
                casting_hook.compute_dtype = torch.float32
                fp32_modulation_hooks += 1

        if "lora_" not in module_name or not hasattr(module, "_diffusers_hook"):
            continue
        casting_hook = module._diffusers_hook.get_hook("layerwise_casting")
        if casting_hook is not None:
            # Diffusers adds layerwise-casting hooks to LoRA leaves loaded after
            # the base model. Their default storage dtype is FP8, so merely
            # changing parameter.data is undone in pre_forward(). LoRAs are
            # small enough to remain BF16 in storage as well as computation.
            casting_hook.storage_dtype = dtype
            casting_hook.compute_dtype = dtype
            fixed_lora_hooks += 1

        # PEFT can clone the base Linear's disk-offload hook into newly-created
        # LoRA Linear leaves. That hook reloads the base layer's cached FP8
        # tensor immediately before F.linear(), undoing parameter.data casts.
        # Adapters are comparatively small, so detach those copied hooks and
        # keep LoRA leaves resident on the accelerator in BF16.
        for hook_name in ("layer_execution_tracker", "lazy_prefetch_group_offloading", "group_offloading"):
            if module._diffusers_hook.get_hook(hook_name) is not None:
                module._diffusers_hook.remove_hook(hook_name, recurse=False)
                detached_offload_hooks += 1
        lora_compute_dtype = (
            torch.float32 if ".img_mod.1." in module_name or ".txt_mod.1." in module_name else dtype
        )
        module.to(device=device, dtype=lora_compute_dtype)

    cast_lora_tensors = 0
    for parameter_name, parameter in pipe.transformer.named_parameters():
        lora_compute_dtype = (
            torch.float32
            if ".img_mod.1." in parameter_name or ".txt_mod.1." in parameter_name
            else dtype
        )
        if (
            "lora_" in parameter_name
            and parameter.is_floating_point()
            and parameter.dtype != lora_compute_dtype
        ):
            parameter.data = parameter.data.to(dtype=lora_compute_dtype)
            cast_lora_tensors += 1
    print(
        f"Pinned {fixed_lora_hooks} LoRA casting hooks, detached {detached_wrapper_hooks} copied "
        f"PEFT-wrapper hooks, {fp32_modulation_hooks} FP32 modulation hooks, "
        f"detached {detached_offload_hooks} copied "
        f"offload hooks, and cast {cast_lora_tensors} tensors "
        f"for {adapter_name} to {dtype} after activation."
    )

    if newly_loaded_adapter:
        # The copied hooks can execute while PEFT is injecting/loading the
        # adapter and corrupt LoRA tensors before we detach them. Reload the
        # original safetensors directly into the now hook-free LoRA leaves.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors

        adapter_path = hf_hub_download(spec["repo"], spec["weights"])
        source_state = load_safetensors(adapter_path, device="cpu")
        target_parameters = dict(pipe.transformer.named_parameters())
        restored_lora_tensors = 0
        missing_lora_tensors = []
        for source_name, source_tensor in source_state.items():
            target_name = peft_parameter_name(source_name, adapter_name)
            target_parameter = target_parameters.get(target_name)
            if target_parameter is None:
                missing_lora_tensors.append(target_name)
                continue
            target_parameter.data.copy_(
                source_tensor.to(device=target_parameter.device, dtype=target_parameter.dtype)
            )
            restored_lora_tensors += 1
        if missing_lora_tensors:
            raise gr.Error(
                f"Could not restore {len(missing_lora_tensors)} LoRA tensors after hook detachment; "
                f"first missing key: {missing_lora_tensors[0]}"
            )
        print(f"Restored {restored_lora_tensors} pristine LoRA tensors from {adapter_path}.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = (
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
    )
    width, height = update_dimensions_on_upload(pil_images[0])

    try:
        result_image = pipe(
            image=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
        ).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example images + prompt + LoRA for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"images": [], "prompt": "", "lora": "", "names": [], "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return {"images": b64_list, "prompt": ex["prompt"], "lora": ex["lora"], "names": names, "status": "ok"}


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: LoRA choices + example card data for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    server_name = os.environ.get("QWEN_SERVER_NAME", "0.0.0.0")
    auth_user = os.environ.get("QWEN_AUTH_USER")
    auth_password = os.environ.get("QWEN_AUTH_PASSWORD")
    auth = (auth_user, auth_password) if auth_user and auth_password else None
    if server_name not in ("127.0.0.1", "localhost") and auth is None:
        raise RuntimeError(
            "QWEN_AUTH_USER and QWEN_AUTH_PASSWORD are required when listening on the LAN."
        )
    app.launch(
        show_error=True,
        mcp_server=True,
        server_name=server_name,
        server_port=int(os.environ.get("QWEN_SERVER_PORT", "7860")),
    )
