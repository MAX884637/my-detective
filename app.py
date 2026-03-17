import streamlit as st
import torch
import torch.nn as nn
import re
import os
import random
import base64
import time
from googletrans import Translator
from gtts import gTTS

# ==========================================
# 1. GPT 模型架構 (確保維度與你的 PTH 匹配)
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps, self.scale = 1e-5, nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x): return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.d_out = cfg["n_heads"], cfg["emb_dim"]
        self.head_dim = self.d_out // self.n_heads
        self.W_query = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer('mask', torch.tril(torch.ones(cfg["context_length"], cfg["context_length"])))
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        attn_scores = (queries @ keys.transpose(-2, -1)) / (keys.shape[-1]**0.5)
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(~mask_bool, -float('inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = (self.dropout(attn_weights) @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att, self.ff = MultiHeadAttention(cfg), FeedForward(cfg)
        self.ln1, self.ln2 = LayerNorm(cfg["emb_dim"]), LayerNorm(cfg["emb_dim"])
    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm, self.out_head = LayerNorm(cfg["emb_dim"]), nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        b, seq_len = in_idx.shape
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len, device=in_idx.device)))
        return self.out_head(self.final_norm(self.trf_blocks(x)))

# ==========================================
# 2. 輔助功能：美化排版、翻譯、語音
# ==========================================
def clean_text(text):
    # 修正常見 NLP 斷詞空格問題
    replacements = [(" ' ", "'"), (" .", "."), (" ,", ","), (" !", "!"), (" ?", "?"), (" n't", "n't"), (" 's", "'s"), (" 'd", "'d"), (" - ", "-")]
    for old, new in replacements: text = text.replace(old, new)
    return text

def translate_to_zh(text):
    try:
        translator = Translator()
        return translator.translate(text, src='en', dest='zh-tw').text
    except: return "【系統提示】翻譯服務繁忙中，請稍後再試。"

def get_audio_html(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("temp_voice.mp3")
        with open("temp_voice.mp3", "rb") as f: data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio controls autoplay src="data:audio/mp3;base64,{b64}">'
    except: return "【系統提示】語音生成失敗。"

# ==========================================
# 3. 展示頁面介面與推理邏輯 (響應式架構)
# ==========================================
st.set_page_config(page_title="J114285102黃政棠 自然語言期末專題：偵探故事屋", layout="wide")

# 初始化會話狀態 (Session State) 以保留故事
if 'generated_en' not in st.session_state: st.session_state['generated_en'] = ""
if 'generated_zh' not in st.session_state: st.session_state['generated_zh'] = ""

# 自定義 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');
    .stApp { background-color: #f7f3e8; }
    .main-title { font-family: 'Special Elite', cursive; color: #1a1a1a; text-align: center; font-size: 3rem; margin-bottom: 0; }
    .sub-title { text-align: center; color: #7f8c8d; font-style: italic; margin-bottom: 2rem; }
    
    .paper-box { background-color: white; padding: 40px; border-radius: 5px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); font-family: 'Georgia', serif; font-size: 22px; line-height: 1.8; color: #2c3e50; min-height: 300px; background-image: linear-gradient(#f1f1f1 1.1em, transparent 1.1em); background-size: 100% 1.5em; border: 1px solid #dcdcdc; }
    .paper-zh-box { background-color: #fdfefe; padding: 40px; border-radius: 5px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); font-family: 'Microsoft JhengHei', sans-serif; font-size: 20px; line-height: 1.8; color: #34495e; min-height: 250px; border-left: 10px solid #3498db; }
    
    .stButton>button { background-color: #2c3e50; color: white; border-radius: 5px; padding: 10px 20px; font-family: 'Georgia'; }
    .stButton>button:hover { background-color: #1a252f; color: #f1c40f; }
    
    .report-header { font-family: ' Special Elite', cursive; color: #16a085; font-size: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    PTH = "gpt_en_pure.pth"
    TXT = r"C:\Users\stan1\OneDrive\Desktop\自然語言學習\PURE_EN_MASTER_DATA.txt"
    with open(TXT, "r", encoding="utf-8") as f: text = f.read()
    pattern = r'([,.::;?_!"()\'\]]|--|\s)'
    tokens = sorted(list(set([t.strip() for t in re.split(pattern, text) if t.strip()])))
    tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {t: i for i, t in enumerate(tokens)}
    inv_vocab = {i: t for t, i in vocab.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(PTH, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model = GPTModel({"vocab_size": state_dict['tok_emb.weight'].shape[0], "context_length": 256, "emb_dim": 384, "n_heads": 6, "n_layers": 6, "drop_rate": 0.1, "qkv_bias": False}).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab, inv_vocab, device

model, vocab, inv_vocab, device = load_resources()

st.markdown('<h1 class="main-title">Detective Story House</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Pure English GPT Model (Loss 0.40)</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 推理設定")
    temp = st.slider("創意溫度 (Creativity)", 0.1, 1.2, 0.7)
    length = st.slider("故事長度 (Tokens)", 50, 400, 150)
    st.divider()
    st.markdown("**專題數據：**\n- Text Characters:~7.9萬\n- Vocab Size: ~5.1萬\n- Emb Dim: 384，6層\n- 文風：經典偵探冒險")
    if st.button("🗑️ 清除當前線索"):
        st.session_state['generated_en'] = ""
        st.session_state['generated_zh'] = ""
        st.rerun()

prompt = st.text_input("🖋️ 輸入線索 (故事開頭)：", value="The detective noticed a strange shadow near the window...")

if st.button("🚀 啟動偵探推理"):
    pattern = r'([,.::;?_!"()\'\]]|--|\s)'
    ids = [vocab.get(t, vocab["<|unk|>"]) for t in re.split(pattern, prompt) if t.strip()]
    input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    
    full_en = prompt
    paper_placeholder = st.empty()
    
    for _ in range(length):
        with torch.no_grad():
            logits = model(input_tensor[:, -256:])
            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat((input_tensor, next_id), dim=1)
            token = inv_vocab.get(next_id.item(), "")
            if token == "<|endoftext|>": break
            
            if token in [".", ",", "!", "?", "'s", "n't"]: full_en += token
            else: full_en += " " + token
            
            # 即時打字效果
            placeholder_text = f'<div class="paper-box">{clean_text(full_en)}▌</div>'
            paper_placeholder.markdown(placeholder_text, unsafe_allow_html=True)
            
    # 存入 Session State
    st.session_state['generated_en'] = clean_text(full_en)
    
    with st.spinner("🕵️‍♂️ 偵探正在翻譯線索..."):
        st.session_state['generated_zh'] = translate_to_zh(st.session_state['generated_en'])
    
    st.rerun()

# 顯示區塊 (保留推理內容)
if st.session_state['generated_en']:
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="report-header">📜 英文寫作</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="paper-box">{st.session_state["generated_en"]}</div>', unsafe_allow_html=True)
        
        # 語音按鈕
        if st.button("🔊 語音朗讀英文稿"):
            st.markdown(get_audio_html(st.session_state['generated_en']), unsafe_allow_html=True)
            
    with col2:
        st.markdown('<p class="report-header">🏮 中文翻譯</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="paper-zh-box">{st.session_state["generated_zh"]}</div>', unsafe_allow_html=True)
        
        # 下載按鈕
        story_content = f"English Story:\n{st.session_state['generated_en']}\n\n中文翻譯:\n{st.session_state['generated_zh']}"
        st.download_button("💾 儲存故事線索 (.txt)", story_content, file_name="ai_detective_story.txt")

st.divider()
st.caption("Final Project Showcase | Pure English GPT Model | Loss 0.40")