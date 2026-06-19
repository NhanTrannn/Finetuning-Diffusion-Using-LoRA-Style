from __future__ import annotations

import gc
import io
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,          # thêm
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,        # thêm
    UniPCMultistepScheduler,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Impressionism LoRA Studio",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --accent: #c2703d;
            --accent-soft: #e9c9a8;
            --accent-deep: #9c5527;
            --teal: #4f8a76;
            --ink: #3a322c;
            --muted: #6a5d50;
        }
        html, body, [class*="css"] {font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;}
        .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1320px;}
        #MainMenu, footer {visibility: hidden;}

        /* ---------- HERO ---------- */
        .hero {
            position: relative;
            padding: 2.2rem 2.4rem;
            border-radius: 26px;
            background: linear-gradient(125deg, #f6ecd9 0%, #efe2d4 38%, #dceae6 100%);
            box-shadow: 0 22px 48px -26px rgba(120, 80, 40, .55);
            overflow: hidden;
            margin-bottom: 1.6rem;
            border: 1px solid rgba(255,255,255,.6);
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,.5) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.5) 1px, transparent 1px);
            background-size: 26px 26px;
            opacity: .25;
            pointer-events: none;
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(460px 240px at 90% -25%, rgba(194,112,61,.25), transparent 60%),
                radial-gradient(380px 260px at 4% 130%, rgba(79,138,118,.22), transparent 60%);
            pointer-events: none;
        }
        .hero-eyebrow {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            color: var(--accent-deep);
            background: rgba(255,255,255,.65);
            border: 1px solid rgba(194,112,61,.3);
            border-radius: 999px;
            padding: .3rem .75rem;
            margin-bottom: .9rem;
        }
        .hero h1 {
            position: relative;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -.6px;
            color: #2c241d;
        }
        .hero p {position: relative; margin: .6rem 0 0 0; color: var(--muted); font-size: 1.04rem; max-width: 62ch; line-height: 1.55;}
        .badge-row {position: relative; margin-top: 1.1rem; display: flex; flex-wrap: wrap; gap: .45rem;}
        .badge {
            display: inline-flex;
            align-items: center;
            gap: .3rem;
            padding: .32rem .85rem;
            border-radius: 999px;
            background: rgba(255,255,255,.75);
            border: 1px solid rgba(194,112,61,.28);
            font-size: .82rem;
            font-weight: 600;
            color: #6a4a30;
            backdrop-filter: blur(6px);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        .badge:hover {transform: translateY(-2px); box-shadow: 0 8px 16px -10px rgba(120,80,40,.5);}

        /* ---------- MODEL CARD ---------- */
        .card {
            position: relative;
            height: 100%;
            padding: 1.25rem 1.3rem;
            border-radius: 18px;
            border: 1px solid rgba(150,120,90,.20);
            border-left: 4px solid var(--accent);
            background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(250,245,238,.7));
            box-shadow: 0 10px 26px -22px rgba(90,60,30,.6);
            transition: transform .18s ease, box-shadow .18s ease;
        }
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 34px -22px rgba(90,60,30,.75);
        }
        .card h4 {margin: 0 0 .3rem 0; font-size: 1.06rem; color: #2c241d; display: flex; align-items: center; gap: .4rem;}
        .card .mono {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: .76rem;
            color: var(--accent);
            background: rgba(194,112,61,.10);
            padding: .14rem .45rem;
            border-radius: 6px;
            display: inline-block;
            margin-bottom: .6rem;
            word-break: break-all;
        }
        .card p {margin: .35rem 0 0 0; font-size: .88rem; color: var(--muted); line-height: 1.5;}

        /* ---------- SECTION TITLE ---------- */
        .sect {
            font-size: 1.2rem;
            font-weight: 700;
            color: #2c241d;
            margin: .2rem 0 .9rem 0;
            padding-left: .65rem;
            border-left: 3px solid var(--accent);
        }
        .sect span {color: var(--accent);}

        /* ---------- TABS ---------- */
        button[data-baseweb="tab"] {
            font-size: .96rem;
            font-weight: 600;
            padding: .5rem 1rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {color: var(--accent-deep);}
        div[data-baseweb="tab-highlight"] {background-color: var(--accent) !important; height: 3px; border-radius: 3px;}

        /* ---------- BUTTONS ---------- */
        button[kind="primary"], .stDownloadButton button {
            border-radius: 12px !important;
            background: linear-gradient(125deg, var(--accent) 0%, var(--accent-deep) 100%) !important;
            border: none !important;
            box-shadow: 0 10px 22px -14px rgba(156,85,39,.7);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        button[kind="primary"]:hover, .stDownloadButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 26px -14px rgba(156,85,39,.85);
        }

        /* ---------- METRICS / ALERTS / CONTAINERS ---------- */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.6);
            border: 1px solid rgba(150,120,90,.18);
            border-radius: 14px;
            padding: .6rem .9rem;
        }
        div[data-testid="stExpander"], div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
        .stAlert {border-radius: 12px;}

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fbf6ee 0%, #f3ece1 100%);
            border-right: 1px solid rgba(150,120,90,.18);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL REGISTRY — CHỈ GIỮ NOTEBOOK HUẤN LUYỆN ĐÃ CÓ
# ============================================================
MODELS = {
    "Realistic Vision V6 + LoRA": {
        "base_model": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "family": "sd15",
        "vae": "stabilityai/sd-vae-ft-mse",
        "controlnet": "lllyasviel/sd-controlnet-canny",
        "notebook": "training_notebook/chonnhan.ipynb",
        "lora_hint": "weight/chonnhan_realistic.safetensors",
        "keywords": ["realistic", "vision", "cn", "chonnhan"],
        "note": "Backbone noVAE nên phải nạp VAE ngoài stabilityai/sd-vae-ft-mse.",
        "icon": "🖼️",
    },
    "Dreamlike Diffusion + LoRA": {
        "base_model": "dreamlike-art/dreamlike-diffusion-1.0",
        "family": "sd15",
        "vae": None,
        "controlnet": "lllyasviel/sd-controlnet-canny",
        "notebook": "training_notebook/pnhat.ipynb",
        "lora_hint": "weight/pnhat_dreamlike.safetensors",
        "keywords": ["dreamlike", "pnhat", "phan", "nhat"],
        "note": "Dùng pipeline Stable Diffusion v1.x với ControlNet Canny.",
        "icon": "💭",
    },
    "SDXL Base 1.0 + LoRA": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl",
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "controlnet": "diffusers/controlnet-canny-sdxl-1.0",
        "notebook": "training_notebook/ducnguyen.ipynb",
        "lora_hint": "weight/ducnguyen_SDXL.safetensors",
        "keywords": ["sdxl", "dn", "ducnguyen"],
        "note": "SDXL phải dùng StableDiffusionXLControlNetPipeline và SDXL ControlNet riêng.",
        "icon": "🚀",
    },
    "Stable Diffusion v1.5 + LoRA": {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "family": "sd15",
        "vae": "stabilityai/sd-vae-ft-mse",
        "controlnet": "lllyasviel/sd-controlnet-canny",
        "notebook": "training_notebook/nhantran-code.ipynb",
        "lora_hint": "weight/nhantran_sdv15.safetensors",
        "keywords": ["sdv1.5", "sdv15", "nhantran"],
        "note": "Backbone SD v1.5 với LoRA custom.",
        "icon": "🎞️",
    },
    "Ghibli + LoRA": {
        "base_model": "nitrosocke/Ghibli-Diffusion",
        "family": "sd15",
        "vae": None,
        "controlnet": "lllyasviel/sd-controlnet-canny",
        "notebook": "training_notebook/DInhLong.ipynb",
        "lora_hint": "weight/dinhlong_Ghibli.safetensors",
        "keywords": ["ghibli", "dinhlong", "dinh", "long"],
        "note": "Ghibli style diffusion model với LoRA custom.",
        "icon": "🍃",
    },
}

# Các kết quả đã có từ notebook huấn luyện.
# Không tự bịa metric khi notebook chưa chạy xong phần đánh giá.
TRAINING_RESULTS = pd.DataFrame(
    [
        {
            "Model": "Realistic Vision V6 + LoRA",
            "Best epoch": "1",
            "Validation loss ↓": 0.188932,
            "FID Base ↓": 161.323196,
            "FID LoRA ↓": 157.434753,
            "KID Base ↓": 0.067046,
            "KID LoRA ↓": 0.065517,
            "Trạng thái": "Held-out style test: 1.306 ảnh",
        },
        {
            "Model": "Dreamlike Diffusion + LoRA",
            "Best epoch": "1",
            "Validation loss ↓": 0.188194,
            "FID Base ↓": 339.563324,
            "FID LoRA ↓": 321.308472,
            "KID Base ↓": 0.064764,
            "KID LoRA ↓": 0.061108,
            "Trạng thái": "Smoke test: 20 ảnh, chưa dùng kết luận chính",
        },
        {
            "Model": "SDXL Base 1.0 + LoRA",
            "Best epoch": "—",
            "Validation loss ↓": np.nan,
            "FID Base ↓": np.nan,
            "FID LoRA ↓": np.nan,
            "KID Base ↓": np.nan,
            "KID LoRA ↓": np.nan,
            "Trạng thái": "Notebook chưa có output đánh giá hoàn chỉnh",
        },
    ]
)

DEFAULT_PROMPT = "a portrait of a woman, skstyle impressionism style painting, soft brush strokes, natural light"
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark, text, artifacts"
OUTPUT_RESOLUTION = 512
NO_WEIGHT_LABEL = "Không dùng weight / chỉ chạy Base"

# Các chế độ cần ảnh đầu vào
IMAGE_INPUT_MODES = (
    "Image-guided (ControlNet Canny + LoRA)",
    "Image-to-image (LoRA)",
)


# ============================================================
# HELPERS
# ============================================================
def runtime_status() -> dict:
    has_cuda = torch.cuda.is_available()
    return {
        "device": "cuda" if has_cuda else "cpu",
        "dtype": torch.float16 if has_cuda else torch.float32,
        "gpu": torch.cuda.get_device_name(0) if has_cuda else "CPU",
    }


def make_canny(image: Image.Image, low: int, high: int, size: int = OUTPUT_RESOLUTION) -> tuple[Image.Image, Image.Image]:
    source = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(source)
    edge = cv2.Canny(array, low, high)
    edge_rgb = np.stack([edge, edge, edge], axis=-1)
    return source, Image.fromarray(edge_rgb)


def save_uploaded_weight(uploaded_weight, model_name: str) -> Optional[Path]:
    if uploaded_weight is None:
        return None
    model_folder = model_name.lower().replace(" ", "_").replace("+", "plus")
    target_dir = Path(tempfile.gettempdir()) / "impressionism_demo_lora" / model_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(uploaded_weight.name).name
    target_path.write_bytes(uploaded_weight.getbuffer())
    return target_path


def find_local_lora_candidates(config: dict) -> list[Path]:
    roots = [Path("."), Path("/kaggle/input"), Path("/kaggle/working")]
    candidates: list[tuple[int, Path]] = []
    seen = set()

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.safetensors"):
            absolute = str(path.resolve())
            if absolute in seen:
                continue
            seen.add(absolute)
            lower = absolute.lower()
            score = sum(keyword.lower() in lower for keyword in config["keywords"])
            if "lora" in lower or "adapter" in lower or score > 0:
                candidates.append((score, path))

    candidates.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return [path for _, path in candidates]


def resolve_default_lora(config: dict, candidates: list[Path]) -> Optional[Path]:
    """Tự động chọn weight đúng model: ưu tiên file khớp keyword, sau đó tới lora_hint."""
    keywords = [k.lower() for k in config["keywords"]]
    for path in candidates:
        if any(k in str(path).lower() for k in keywords):
            return path
    hint = Path(config["lora_hint"].replace("\\", "/"))
    if hint.exists() and hint.is_file():
        return hint
    return None


def pil_download_button(image: Image.Image, filename: str, label: str) -> None:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
        use_container_width=True,
    )


@st.cache_resource(show_spinner=False, max_entries=3)
def load_base_pipeline(model_name: str, inference_mode: str, low_vram: bool):
    """Cache pipeline chưa gắn LoRA; mỗi lần Generate sẽ nạp adapter được chọn."""
    config = MODELS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    common_kwargs = {"torch_dtype": dtype}

    if config["vae"]:
        common_kwargs["vae"] = AutoencoderKL.from_pretrained(
            config["vae"],
            torch_dtype=dtype,
        )

    if inference_mode == "Image-guided (ControlNet Canny + LoRA)":
        controlnet = ControlNetModel.from_pretrained(
            config["controlnet"],
            torch_dtype=dtype,
        )
        common_kwargs["controlnet"] = controlnet
        pipeline_class = (
            StableDiffusionXLControlNetPipeline
            if config["family"] == "sdxl"
            else StableDiffusionControlNetPipeline
        )
    elif inference_mode == "Image-to-image (LoRA)":
        pipeline_class = (
            StableDiffusionXLImg2ImgPipeline
            if config["family"] == "sdxl"
            else StableDiffusionImg2ImgPipeline
        )
    else:
        pipeline_class = (
            StableDiffusionXLPipeline
            if config["family"] == "sdxl"
            else StableDiffusionPipeline
        )

    if config["family"] == "sdxl" and device == "cuda":
        common_kwargs.update({"variant": "fp16", "use_safetensors": True})

    pipe = pipeline_class.from_pretrained(config["base_model"], **common_kwargs)

    # Tắt safety checker (NSFW filter hay báo nhầm → trả ảnh đen)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if device == "cuda" and low_vram:
        try:
            pipe.enable_model_cpu_offload()
        except (ImportError, AttributeError):
            st.warning(
                "Không bật được model CPU offload (accelerate < 0.17.0). "
                "Đang chuyển pipeline thẳng lên GPU — nâng cấp bằng `pip install -U accelerate` để tiết kiệm VRAM."
            )
            pipe.to(device)
    else:
        pipe.to(device)

    return pipe


def clear_loaded_lora(pipe) -> None:
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass


def load_selected_lora(pipe, lora_path: Path, lora_scale: float) -> None:
    if not lora_path.exists() or not lora_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file LoRA: {lora_path}")

    clear_loaded_lora(pipe)
    pipe.load_lora_weights(
        str(lora_path.parent),
        weight_name=lora_path.name,
        adapter_name="demo_lora",
    )
    try:
        pipe.set_adapters(["demo_lora"], adapter_weights=[float(lora_scale)])
    except (AttributeError, TypeError):
        if abs(lora_scale - 1.0) > 1e-6:
            st.warning("Phiên bản diffusers hiện tại không hỗ trợ điều chỉnh adapter scale động; đang dùng scale mặc định 1.0.")


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str,
    mode: str,
    canny_image: Optional[Image.Image],
    steps: int,
    guidance: float,
    control_scale: float,
    seed: int,
    init_image: Optional[Image.Image] = None,   # thêm
    strength: float = 0.6,                       # thêm
) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))
    arguments = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "generator": generator,
    }
    if mode == "Image-guided (ControlNet Canny + LoRA)":
        arguments["image"] = canny_image
        arguments["controlnet_conditioning_scale"] = float(control_scale)
        arguments["height"] = OUTPUT_RESOLUTION
        arguments["width"] = OUTPUT_RESOLUTION
    elif mode == "Image-to-image (LoRA)":
        arguments["image"] = init_image
        arguments["strength"] = float(strength)
        # img2img lấy kích thước từ ảnh init, KHÔNG truyền height/width
    else:  # text-to-image
        arguments["height"] = OUTPUT_RESOLUTION
        arguments["width"] = OUTPUT_RESOLUTION

    with torch.inference_mode():
        return pipe(**arguments).images[0]


# ============================================================
# INTERFACE
# ============================================================
status = runtime_status()

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">✨ LoRA fine-tuning studio</div>
        <h1>🎨 Impressionism LoRA Studio</h1>
        <p>Trình bày kết quả huấn luyện và so sánh trực tiếp <b>Base</b> với <b>LoRA fine-tuned</b>
        qua ControlNet Canny — weight được gắn tự động cho từng model.</p>
        <div class="badge-row">
            <span class="badge">🖼️ Realistic Vision</span>
            <span class="badge">💭 Dreamlike</span>
            <span class="badge">🚀 SDXL</span>
            <span class="badge">🎞️ SD v1.5</span>
            <span class="badge">🍃 Ghibli</span>
            <span class="badge">🧬 LoRA + ControlNet</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Môi trường")
    st.metric("Thiết bị", status["gpu"])
    if status["device"] == "cuda":
        st.success("CUDA sẵn sàng.")
    else:
        st.warning("Không có GPU: xem kết quả được, nhưng sinh ảnh sẽ rất chậm.")

    st.divider()
    st.caption(f"Số model trong registry: **{len(MODELS)}**")
    if st.button("🧹 Xóa cache pipeline", use_container_width=True):
        load_base_pipeline.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        st.success("Đã xóa cache.")

    st.divider()
    st.caption("🎨 Impressionism LoRA Studio")

overview_tab, results_tab, inference_tab, config_tab = st.tabs(
    ["🏠 Tổng quan", "📊 Kết quả huấn luyện", "🖼️ Demo inference", "🧩 Cấu hình model"]
)

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with overview_tab:
    st.markdown('<div class="sect">Các model đang được <span>demo</span></div>', unsafe_allow_html=True)

    model_items = list(MODELS.items())
    for start in range(0, len(model_items), 3):
        cards = st.columns(3)
        for column, (model_name, config) in zip(cards, model_items[start:start + 3]):
            with column:
                st.markdown(
                    f"""
                    <div class="card">
                        <h4><span>{config['icon']}</span> {model_name}</h4>
                        <span class="mono">{config['base_model']}</span>
                        <p><b>Notebook:</b> {config['notebook']}</p>
                        <p>{config['note']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")

    st.divider()
    st.markdown('<div class="sect">Luồng <span>xử lý</span></div>', unsafe_allow_html=True)
    flow_a, flow_b, flow_c = st.columns(3)
    flow_a.markdown(
        "**1 · Training**\n\nLoRA được fine-tune để học phong cách "
        "`skstyle impressionism style painting`."
    )
    flow_b.markdown(
        "**2 · Đánh giá**\n\nDùng Validation Loss, FID và KID; CLIPScore "
        "có thể bổ sung sau khi cả ba model chạy thống nhất."
    )
    flow_c.markdown(
        "**3 · Inference**\n\nẢnh vào → Canny/img2img → backbone + LoRA → "
        "so sánh **Base** vs **LoRA** cùng prompt/seed."
    )

    st.info(
        "Lưu ý: notebook Realistic Vision có bảng metric ghi nhầm nhãn Dreamlike; "
        "trong app này kết quả đã được hiển thị đúng dưới tên Realistic Vision."
    )

# ============================================================
# TAB 2 — RESULTS
# ============================================================
with results_tab:
    st.markdown('<div class="sect">Kết quả huấn luyện <span>đã có</span></div>', unsafe_allow_html=True)
    st.dataframe(
        TRAINING_RESULTS,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Validation loss ↓": st.column_config.NumberColumn(format="%.6f"),
            "FID Base ↓": st.column_config.NumberColumn(format="%.3f"),
            "FID LoRA ↓": st.column_config.NumberColumn(format="%.3f"),
            "KID Base ↓": st.column_config.NumberColumn(format="%.6f"),
            "KID LoRA ↓": st.column_config.NumberColumn(format="%.6f"),
        },
    )

    selected_result_model = st.selectbox(
        "Xem biểu đồ của model",
        list(MODELS.keys()),
        key="result_model",
    )
    matched = TRAINING_RESULTS[TRAINING_RESULTS["Model"] == selected_result_model]
    if matched.empty:
        st.info("Model này chưa được thêm vào bảng kết quả huấn luyện.")
    else:
        row = matched.iloc[0]
        if pd.isna(row["FID Base ↓"]):
            st.warning("Notebook này hiện chưa có output FID/KID hoàn chỉnh; không hiển thị số liệu giả.")
        else:
            fid_delta = float(row["FID Base ↓"] - row["FID LoRA ↓"])
            kid_delta = float(row["KID Base ↓"] - row["KID LoRA ↓"])

            m1, m2, m3 = st.columns(3)
            m1.metric("FID (LoRA)", f"{row['FID LoRA ↓']:.3f}", delta=f"-{fid_delta:.3f}", delta_color="inverse")
            m2.metric("KID (LoRA)", f"{row['KID LoRA ↓']:.6f}", delta=f"-{kid_delta:.6f}", delta_color="inverse")
            m3.metric("Validation loss", f"{row['Validation loss ↓']:.6f}")

            chart_data = pd.DataFrame(
                {
                    "Phiên bản": ["Base", "LoRA"],
                    "FID ↓": [row["FID Base ↓"], row["FID LoRA ↓"]],
                    "KID ↓": [row["KID Base ↓"], row["KID LoRA ↓"]],
                }
            )
            fid_col, kid_col = st.columns(2)
            with fid_col:
                st.markdown("#### FID — thấp hơn là tốt hơn")
                st.bar_chart(chart_data.set_index("Phiên bản")[["FID ↓"]], color="#c2703d")
            with kid_col:
                st.markdown("#### KID — thấp hơn là tốt hơn")
                st.bar_chart(chart_data.set_index("Phiên bản")[["KID ↓"]], color="#5a8c78")

# ============================================================
# TAB 3 — LIVE INFERENCE
# ============================================================
with inference_tab:
    st.markdown('<div class="sect">Sinh ảnh & so sánh <span>Base vs LoRA</span></div>', unsafe_allow_html=True)
    settings_col, output_col = st.columns([1, 1.55])

    with settings_col:
        model_name = st.selectbox("Model", list(MODELS.keys()), key="inference_model")
        config = MODELS[model_name]

        inference_mode = st.radio(
            "Chế độ",
            [
                "Image-guided (ControlNet Canny + LoRA)",
                "Image-to-image (LoRA)",
                "Text-to-image (LoRA)",
            ],
        )

        prompt = st.text_area("Prompt", DEFAULT_PROMPT, height=88)
        negative_prompt = st.text_input("Negative prompt", DEFAULT_NEGATIVE_PROMPT)

        st.markdown("#### LoRA weight")

        # --- TỰ ĐỘNG DÒ & CHỌN WEIGHT ĐÚNG MODEL ---
        candidates = find_local_lora_candidates(config)
        default_lora = resolve_default_lora(config, candidates)

        uploaded_weight = st.file_uploader(
            "Ghi đè bằng file `.safetensors` (tuỳ chọn)",
            type=["safetensors"],
            key=f"lora_{model_name}",
        )
        uploaded_path = save_uploaded_weight(uploaded_weight, model_name)

        option_paths = [str(p) for p in candidates]
        if default_lora is not None and str(default_lora) not in option_paths:
            option_paths.insert(0, str(default_lora))
        path_options = [NO_WEIGHT_LABEL] + option_paths

        auto_index = 0
        if default_lora is not None and str(default_lora) in path_options:
            auto_index = path_options.index(str(default_lora))

        if uploaded_path:
            chosen_weight_path = uploaded_path
            st.success(f"Dùng weight vừa upload: `{uploaded_path.name}`")
        else:
            chosen_option = st.selectbox(
                "Weight LoRA (đã tự động chọn sẵn nếu tìm thấy)",
                path_options,
                index=auto_index,
            )
            chosen_weight_path = None if chosen_option == NO_WEIGHT_LABEL else Path(chosen_option)
            if chosen_weight_path is not None and auto_index != 0 and chosen_option == path_options[auto_index]:
                st.success(f"Tự động gắn weight: `{chosen_weight_path.name}`")
            elif default_lora is None:
                st.caption("Chưa dò được weight tự động — có thể upload hoặc nhập đường dẫn bên dưới.")

        manual_weight = st.text_input(
            "Hoặc nhập đường dẫn weight thủ công",
            value="",
            placeholder=config["lora_hint"],
        )
        if manual_weight.strip():
            chosen_weight_path = Path(manual_weight.strip())

        uploaded_image = None
        if inference_mode in IMAGE_INPUT_MODES:
            uploaded_image = st.file_uploader(
                "Ảnh đầu vào",
                type=["jpg", "jpeg", "png", "webp"],
                key="source_image",
            )

        with st.expander("Thông số sinh ảnh", expanded=True):
            steps = st.slider("Inference steps", 10, 50, 25)
            guidance = st.slider("Guidance scale", 1.0, 12.0, 7.5, 0.5)
            control_scale = st.slider(
                "ControlNet scale",
                0.1,
                2.0,
                0.85,
                0.05,
                disabled=inference_mode != "Image-guided (ControlNet Canny + LoRA)",
            )
            lora_scale = st.slider("LoRA scale", 0.0, 2.0, 1.0, 0.05)
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42)
            canny_low = st.slider("Canny low", 0, 255, 100, disabled=inference_mode != "Image-guided (ControlNet Canny + LoRA)")
            canny_high = st.slider("Canny high", 0, 255, 200, disabled=inference_mode != "Image-guided (ControlNet Canny + LoRA)")
            compare_base = st.checkbox("Sinh cả Base để so sánh", value=True)
            low_vram = st.checkbox("Low VRAM mode", value=True)
            strength = st.slider(
                "Strength (img2img — thấp = giữ ảnh gốc)",
                0.1, 1.0, 0.6, 0.05,
                disabled=inference_mode != "Image-to-image (LoRA)",
            )

        generate_clicked = st.button("✨ Sinh ảnh demo", type="primary", use_container_width=True)

    input_image = None
    canny_image = None
    init_image = None

    with output_col:
        if uploaded_image is not None:
            source = Image.open(uploaded_image)
            if inference_mode == "Image-to-image (LoRA)":
                init_image = source.convert("RGB").resize(
                    (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), Image.Resampling.LANCZOS
                )
                st.image(init_image, caption="Input (img2img)", use_container_width=True)
            else:
                input_image, canny_image = make_canny(source, canny_low, canny_high)
                preview_a, preview_b = st.columns(2)
                preview_a.image(input_image, caption="Input", use_container_width=True)
                preview_b.image(canny_image, caption="Canny control", use_container_width=True)
        else:
            st.info("Tải ảnh đầu vào (ControlNet / img2img) hoặc bấm sinh ảnh ngay với chế độ Text-to-image.")

        if generate_clicked:
            if inference_mode in IMAGE_INPUT_MODES and uploaded_image is None:
                st.error("Chế độ này cần tải ảnh đầu vào.")
                st.stop()

            if chosen_weight_path is not None and not chosen_weight_path.exists():
                st.error(f"Không tìm thấy LoRA weight tại: `{chosen_weight_path}`")
                st.stop()

            try:
                with st.spinner("Đang nạp pipeline..."):
                    pipe = load_base_pipeline(model_name, inference_mode, low_vram)

                results = {}
                if compare_base or chosen_weight_path is None:
                    clear_loaded_lora(pipe)
                    with st.spinner("Đang sinh ảnh Base..."):
                        results["Base model"] = generate_image(
                            pipe, prompt, negative_prompt, inference_mode,
                            canny_image, steps, guidance, control_scale, int(seed),
                            init_image=init_image, strength=strength,
                        )

                if chosen_weight_path is not None:
                    with st.spinner("Đang nạp LoRA và sinh ảnh fine-tuned..."):
                        load_selected_lora(pipe, chosen_weight_path, lora_scale)
                        results["LoRA fine-tuned"] = generate_image(
                            pipe, prompt, negative_prompt, inference_mode,
                            canny_image, steps, guidance, control_scale, int(seed),
                            init_image=init_image, strength=strength,
                        )

                result_columns = st.columns(len(results))
                for column, (label, image) in zip(result_columns, results.items()):
                    with column:
                        st.image(image, caption=label, use_container_width=True)
                        safe_name = label.lower().replace(" ", "_") + ".png"
                        pil_download_button(image, safe_name, f"⬇️ Tải {label}")

                st.success(f"Hoàn tất inference bằng {model_name}.")

            except Exception as error:
                st.exception(error)
                st.error(
                    "Kiểm tra lại GPU, Internet/Hugging Face model, và bảo đảm LoRA thuộc đúng backbone đã chọn."
                )

# ============================================================
# TAB 4 — CONFIG
# ============================================================
with config_tab:
    st.markdown('<div class="sect">Registry model & điểm cần <span>giữ đúng</span></div>', unsafe_allow_html=True)
    registry_rows = []
    for name, config in MODELS.items():
        registry_rows.append(
            {
                "Model": f"{config['icon']} {name}",
                "Base model": config["base_model"],
                "VAE": config["vae"] or "Mặc định",
                "ControlNet": config["controlnet"],
                "Notebook": config["notebook"],
                "Weight mặc định": config["lora_hint"],
            }
        )
    st.dataframe(pd.DataFrame(registry_rows), hide_index=True, use_container_width=True)

    st.markdown(
        """
        **Nguyên tắc bắt buộc**
        - LoRA của Realistic Vision chỉ nạp vào Realistic Vision.
        - LoRA của Dreamlike chỉ nạp vào Dreamlike.
        - LoRA của SDXL chỉ nạp vào SDXL.
        - SDXL không được chạy bằng `StableDiffusionControlNetPipeline`; app đã tách sang `StableDiffusionXLControlNetPipeline`.
        - Không `fuse_lora()` trong demo tương tác vì cần thay đổi LoRA scale và sinh Base/LoRA độc lập.
        - Auto-load chỉ chọn weight có **tên khớp keyword của đúng model**, nên không thể gắn nhầm LoRA của model khác.
        """
    )
