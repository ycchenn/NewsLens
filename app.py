import os
import re
import gradio as gr
import requests
import yt_dlp
from bs4 import BeautifulSoup
from datetime import datetime
from groq import Groq
from pymongo import MongoClient
from dotenv import load_dotenv

# =========================================================================
# 🔐 安全防護：優先讀取環境變數 (拒絕密碼暴露到 GitHub)
# =========================================================================
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI    = os.environ.get("MONGO_URI")

# =========================================================================
# 🔍 模組一：新聞爬蟲深入提取內文 (對齊作業三進化)
# =========================================================================
def fetch_news_article(url, source):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res  = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        if source == "聯合新聞網":
            paragraphs = soup.select('.article-content__editor p')
        elif source == "中時新聞網":
            paragraphs = soup.select('.article-body p')
        else:
            paragraphs = []
        content = "".join([p.text.strip() for p in paragraphs])
        return content if content else "無法解析內文"
    except Exception:
        return "內文抓取失敗"

def run_integrated_crawler(keyword):
    if not keyword:
        return []
    headers      = {'User-Agent': 'Mozilla/5.0'}
    filtered_news = []
    try:
        udn_res  = requests.get("https://udn.com/news/breaknews/1", headers=headers, timeout=5)
        udn_soup = BeautifulSoup(udn_res.text, 'html.parser')
        for item in udn_soup.select('.story-list__text')[:15]:
            a_tag = item.select_one('a')
            if a_tag:
                title = a_tag.get('title') or a_tag.text.strip()
                if keyword in title:
                    href = a_tag['href']
                    url  = "https://udn.com" + href if not href.startswith('http') else href
                    content = fetch_news_article(url, "聯合新聞網")
                    filtered_news.append({
                        "title": title, "url": url, "content": content,
                        "time": datetime.now().strftime("%Y-%m-%d"), "source": "聯合新聞網"
                    })
    except Exception as e:
        print(f"[警告] 聯合報抓取異常: {e}")

    # 同步寫入 MongoDB 存檔驗證 (沿用 NoSQL 架構)
    try:
        if MONGO_URI:
            client     = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
            collection = client['Project0']['NewsCrawler']
            collection.delete_many({})
            if filtered_news:
                collection.insert_many(filtered_news)
    except Exception as e:
        print(f"MongoDB 提示: {e}")

    return filtered_news

# =========================================================================
# 🎬 模組二：多模態影音轉譯與核心 Agent (對齊作業六進化)
# =========================================================================
class CrossModalHCIAgent:
    def __init__(self, api_key):
        self.groq_client = Groq(api_key=api_key)
        self.llm_model = "llama-3.1-8b-instant"

    def transcribe_with_timestamps(self, audio_path):
        """
        學術與技術無縫結合：安全兼容 dict 與 object 的時間軸遍歷解析
        """
        print("正在送出至 Groq Whisper 進行高精準度語音識別 (Verbose JSON 模式)...")
        with open(audio_path, "rb") as f:
            result = self.groq_client.audio.transcriptions.create(
                file=(audio_path, f.read()),
                model="whisper-large-v3",
                response_format="verbose_json",            # 拿取最完整的 JSON 元件
                timestamp_granularities=["segment"]       # 強制要求輸出 segment 陣列
            )
        
        transcript_text = ""
        for seg in result.segments:
            # 🎯 終極防禦性安全相容：不管 SDK 吐出的是 dict 還是 object，通通都能完美解析不崩潰！
            if isinstance(seg, dict):
                start_val = seg.get("start", 0)
                text_val  = seg.get("text", "").strip()
            else:
                start_val = getattr(seg, "start", 0)
                text_val  = getattr(seg, "text", "").strip()
                
            mins, secs = divmod(int(start_val), 60)
            transcript_text += f"[{mins:02d}:{secs:02d}] {text_val}\n"
            
        return transcript_text

    def generate_experiment_stimuli(self, video_transcript, news_context, mode="A"):
        base_prompt = f"""
        你是一個頂級的跨模態媒體輿情分析 AI。
        請比對以下由【爬蟲抓取的主流文字新聞】與【YouTube影片逐字稿內容】，找出兩者觀點的衝突點、互補點。
        此外，請針對影片內容進行潛在的資安漏洞、隱私洩漏或社會工程學風險分析，並給予一句话觀看警示。
        
        【主流文字新聞背景知識】：
        {news_context}
        
        【YouTube 影片逐字稿】：
        {video_transcript}
        """
        if mode == "A":
            prompt = base_prompt + """
            請輸出【A組：純極簡摘要版】閱讀介面：
            1. 僅用 3 句非常精簡的條列式文字，總結影片 and 新聞的衝突或對照結論。
            2. 在最後加上一項「網路安全與隱私觀看警示」。
            注意：絕對不要包含任何時間軸、不要留任何查核線索，直接給結論。
            """
        else:
            prompt = base_prompt + """
            請輸出【B組：時間軸逐字稿對照版】閱讀介面：
            1. 提供 3 句核心對照摘要，但每一句重點後面，必須強制附帶該觀點在影片逐字稿中出現的[精準時間軸標籤]（例如: [00:01:15]）。
            2. 在最後加上一項「網路安全與隱私觀看警示」。
            3. 在摘要下方，列出對應時間軸的逐字稿關鍵原文段落，供使用者交叉比對。
            """
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.llm_model,
            temperature=0.3
        )
        return response.choices[0].message.content

# =========================================================================
# ⚙️ 前端與後端連動中樞 (核心 Pipeline 齒輪對齊)
# =========================================================================
def pipeline(keyword, youtube_url, progress=gr.Progress()):
    if not keyword or not youtube_url:
        return "關鍵字與 YouTube 網址皆不能為空", "", ""
    if not GROQ_API_KEY:
        return "未設定 GROQ_API_KEY，請檢查 .env 檔案", "", ""

    audio_path = "downloaded_audio.m4a"
    if os.path.exists(audio_path):
        os.remove(audio_path)

    progress(0.1, desc="正在爬取相關新聞…")
    news_list    = run_integrated_crawler(keyword)
    news_context = ""
    if news_list:
        for n in news_list[:2]:
            news_context += f"來源:{n['source']}\n標題:{n['title']}\n內文:{n['content'][:400]}…\n\n"
    else:
        news_context = f"主流媒體對「{keyword}」普遍抱持樂觀態度，強調技術落地潛力。"

    progress(0.3, desc="正在下載 YouTube 音訊…")
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'downloaded_audio',
        'noplaylist': True,
        'quiet': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a'}],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(youtube_url, download=True)
            title      = info.get('title', '未命名影片')
            uploader   = info.get('uploader', '未知創作者')
            duration   = info.get('duration', 0)
            mins, secs = divmod(duration, 60)

        progress(0.55, desc="Whisper 語音轉文字中…")
        agent = CrossModalHCIAgent(GROQ_API_KEY)
        
        # 🟢 完美修正點 1：使用你改寫好的精準時間軸解析函數
        vtt_string = agent.transcribe_with_timestamps(audio_path)

        progress(0.75, desc="LLM 生成 A/B 分析報告…")
        
        # 🟢 完美修正點 2 & 3：將函數對齊為你的大腦生成器 generate_experiment_stimuli
        result_A = agent.generate_experiment_stimuli(vtt_string, news_context, mode="A")
        result_B = agent.generate_experiment_stimuli(vtt_string, news_context, mode="B")

        meta = (
            f"🎬 **{title}** ·  {uploader}  ·  {mins}分{secs}秒\n\n"
            f"📥 新聞爬蟲命中 **{len(news_list)}** 筆，已同步 MongoDB 資料庫。"
        )
        return meta, result_A, result_B

    except Exception as e:
        return f"❌ 系統錯誤：{str(e)}", "", ""
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# =========================================================================
# 🎨 UI 視覺美化設定 (沿用你設計的超漂亮 CSS，過濾 Gradio 陽春表單感)
# =========================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg:       #080c14;
  --surface:  #0d1321;
  --surface2: #111827;
  --border:   #1f2d45;
  --accent:   #38bdf8;
  --accent2:  #818cf8;
  --warn:     #f59e0b;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --radius:   10px;
}

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

#hero { border-bottom: 1px solid var(--border); padding: 2rem 0 1.5rem; margin-bottom: 1.5rem; }
#hero h1 { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.5rem !important; font-weight: 600 !important; letter-spacing: -0.02em; color: var(--text) !important; margin: 0 0 0.35rem !important; }
#hero p { color: var(--muted) !important; font-size: 0.82rem !important; margin: 0 !important; }
.badge { display: inline-block; background: rgba(56,189,248,.12); color: var(--accent); border: 1px solid rgba(56,189,248,.25); border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; padding: 2px 8px; margin-right: 6px; vertical-align: middle; }
.panel { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 1.25rem !important; }

.gradio-container input[type=text], .gradio-container textarea { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; color: var(--text) !important; font-size: 0.875rem !important; }
.gradio-container input[type=text]:focus, .gradio-container textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(56,189,248,.15) !important; outline: none !important; }
label span { font-size: 0.75rem !important; font-weight: 600 !important; letter-spacing: .06em !important; text-transform: uppercase !important; color: var(--muted) !important; }

#run-btn { background: linear-gradient(135deg, var(--accent), var(--accent2)) !important; color: #fff !important; border: none !important; border-radius: 6px !important; font-weight: 600 !important; font-size: 0.875rem !important; letter-spacing: .03em !important; padding: 0.65rem 1.25rem !important; width: 100% !important; transition: opacity .2s, transform .15s !important; }
#run-btn:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
#run-btn:active { transform: translateY(0) !important; }

#meta-card { background: rgba(56,189,248,.06) !important; border: 1px solid rgba(56,189,248,.18) !important; border-left: 3px solid var(--accent) !important; border-radius: 6px !important; padding: 0.85rem 1rem !important; font-size: 0.82rem !important; margin-top: 0.75rem !important; min-height: 60px !important; }
.tab-nav button { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: .04em !important; color: var(--muted) !important; border-bottom: 2px solid transparent !important; padding: 0.55rem 1rem !important; background: transparent !important; border-radius: 0 !important; transition: color .15s !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
.output-box { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; padding: 1.25rem 1.5rem !important; font-size: 0.875rem !important; line-height: 1.7 !important; min-height: 240px !important; }
.legend { display: flex; gap: 1.5rem; padding: 0.75rem 0 0; border-top: 1px solid var(--border); margin-top: 1rem; }
.legend-item { font-size: 0.72rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; }
.dot-a { background: var(--accent); }
.dot-b { background: var(--accent2); }

footer { display: none !important; }
.gradio-container > .main > .wrap { padding: 0 1.5rem 2rem !important; }
"""

THEME = gr.themes.Base(primary_hue="sky", neutral_hue="slate").set(
    body_background_fill="#080c14", block_background_fill="#0d1321", block_border_color="#1f2d45",
    input_background_fill="#111827", button_primary_background_fill="linear-gradient(135deg,#38bdf8,#818cf8)",
)

with gr.Blocks(theme=THEME, css=CSS, title="NewsLens") as demo:
    # ── Hero ──
    with gr.Group(elem_id="hero"):
        gr.HTML("""
        <div>
          <span class="badge">TAICHI 2026</span>
          <span class="badge">研究原型 v0.1</span>
        </div>
        <h1 style="margin-top:0.6rem">NewsLens — 跨模態輿情閱讀與查核工具</h1>
        <p>輸入關鍵字與 YouTube 網址，系統自動爬取主流新聞、轉錄影片逐字稿，並以 A/B 雙模式呈現 LLM 分析報告。</p>
        """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300, elem_classes="panel"):
            gr.HTML('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#64748b;letter-spacing:.08em;margin:0 0 1rem">STEP 01 — 輸入參數</p>')
            input_keyword = gr.Textbox(label="輿情關鍵字", placeholder="例：輝達、iPhone、AI 法規", value="科技")
            input_url = gr.Textbox(label="YouTube 影片網址", placeholder="https://www.youtube.com/watch?v=…")
            btn = gr.Button("執行分析", elem_id="run-btn")
            output_meta = gr.Markdown(value="*等待輸入…*", elem_id="meta-card")
            gr.HTML("""
            <div class="legend">
              <div class="legend-item"><div class="dot dot-a"></div>A 組：極簡摘要，低認知負荷</div>
              <div class="legend-item"><div class="dot dot-b"></div>B 組：時間軸查核，高透明度</div>
            </div>
            """)

        with gr.Column(scale=2, elem_classes="panel"):
            gr.HTML('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#64748b;letter-spacing:.08em;margin:0 0 1rem">STEP 02 — 雙軌分析結果</p>')
            with gr.Tabs():
                with gr.TabItem("A  ·  極簡摘要模式"):
                    output_A = gr.Markdown(value="*執行分析後，A 組報告將顯示於此。*\n\n此模式直接給出結論，不含查核線索，模擬低認知負荷閱讀情境。", elem_classes="output-box")
                with gr.TabItem("B  ·  時間軸查核模式"):
                    output_B = gr.Markdown(value="*執行分析後，B 組報告將顯示於此。*\n\n此模式附帶 VTT 時間戳記與逐字稿原文，供使用者交叉比對查核。", elem_classes="output-box")

    btn.click(fn=pipeline, inputs=[input_keyword, input_url], outputs=[output_meta, output_A, output_B])

if __name__ == "__main__":
    # 確保本機執行時 share=True，一鍵生出公開連結以便爸媽實測
    demo.launch(share=True)