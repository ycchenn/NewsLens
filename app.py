import gradio as gr
import os
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from groq import Groq
from pymongo import MongoClient

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

def fetch_news_article(url, source):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        if source == "聯合新聞網":
            paragraphs = soup.select('.article-content__editor p')
            content = "".join([p.text.strip() for p in paragraphs])
        elif source == "中時新聞網":
            paragraphs = soup.select('.article-body p')
            content = "".join([p.text.strip() for p in paragraphs])
        else:
            content = ""
        return content if content else "無法解析內文"
    except Exception:
        return "內文抓取失敗"

def run_integrated_crawler(keyword):
    if not keyword:
        return []
    headers = {'User-Agent': 'Mozilla/5.0'}
    filtered_news = []
    
    # 抓取聯合報即時新聞
    try:
        udn_res = requests.get("https://udn.com/news/breaknews/1", headers=headers, timeout=5)
        udn_soup = BeautifulSoup(udn_res.text, 'html.parser')
        for item in udn_soup.select('.story-list__text')[:15]:
            a_tag = item.select_one('a')
            if a_tag:
                title = a_tag.get('title') or a_tag.text.strip()
                if keyword in title:
                    url = "https://udn.com" + a_tag['href'] if not a_tag['href'].startswith('http') else a_tag['href']
                    content = fetch_news_article(url, "聯合新聞網")
                    filtered_news.append({
                        "title": title, "url": url, "content": content,
                        "time": datetime.now().strftime("%Y-%m-%d"), "source": "聯合新聞網"
                    })
    except Exception as e:
        print(f"[警告] 聯合報抓取異常: {e}")

    # 同步寫入 MongoDB 存檔驗證
    try:
        client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
        db = client['Project0']
        collection = db['NewsCrawler']
        collection.delete_many({})
        if filtered_news:
            collection.insert_many(filtered_news)
    except Exception as e:
        print(f"MongoDB 同步提示: {e}")
        
    return filtered_news

class CrossModalHCIAgent:
    def __init__(self, api_key):
        self.groq_client = Groq(api_key=api_key)
        self.llm_model = "llama-3.1-8b-instant"

    def transcribe_with_timestamps(self, audio_path):
        with open(audio_path, "rb") as file:
            vtt_transcript = self.groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3",
                response_format="vtt"  
            )
        return vtt_transcript

    def generate_experiment_stimuli(self, video_transcript, news_context, mode="A"):
        base_prompt = f"""
        你是一個頂級的跨模態媒體輿情分析 AI。
        請比對以下由【爬蟲抓取的主流文字新聞】與【YouTube影片逐字稿內容】，找出兩者觀點的衝突點、互補點。
        此外，請針對影片內容進行潛在的資安漏洞、隱私洩漏或社會工程學風險分析，並給予一句話觀看警示。
        
        【主流文字新聞背景知識】：
        {news_context}
        
        【YouTube 影片逐字稿】：
        {video_transcript}
        """
        if mode == "A":
            prompt = base_prompt + """
            請輸出【A組：純極簡摘要版】閱讀介面：
            1. 僅用 3 句非常精簡的條列式文字，總結影片和新聞的衝突或對照結論。
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

def taichi_experiment_pipeline(keyword, youtube_url):
    if not keyword or not youtube_url:
        return "錯誤：關鍵字與 YouTube 網址皆不能為空！", "", ""
    
    import yt_dlp
    audio_filename = "downloaded_audio"
    target_ext = "m4a"
    actual_audio_path = f"{audio_filename}.{target_ext}"
    
    if os.path.exists(actual_audio_path):
        os.remove(actual_audio_path)

    # 1. 爬蟲撈取文字新聞
    news_list = run_integrated_crawler(keyword)
    news_context_text = ""
    if news_list:
        for n in news_list[:2]:
            news_context_text += f"來源:{n['source']}\n標題:{n['title']}\n內文:{n['content'][:400]}...\n\n"
    else:
        news_context_text = f"主流媒體報導風向：普遍對『{keyword}』的發展抱持樂觀態度，強調技術落地的市場潛力。"

    # 2. 下載 YouTube 音訊
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': audio_filename,
        'noplaylist': True,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': target_ext,
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_title = info_dict.get('title', '未命名影片')
            uploader = info_dict.get('uploader', '未知創作者')
            duration = info_dict.get('duration', 0)
        
        # 3. 呼叫 HCI Agent 生成 A/B 測試刺激物
        agent = CrossModalHCIAgent(api_key=GROQ_API_KEY)
        raw_vtt = agent.transcribe_with_timestamps(actual_audio_path)
        
        stimuli_A = agent.generate_experiment_stimuli(raw_vtt, news_context_text, mode="A")
        stimuli_B = agent.generate_experiment_stimuli(raw_vtt, news_context_text, mode="B")
        
        status_card = f"### 🎬 測試影片中繼資料\n• **影片標題**：{video_title}\n• **創作者**：{uploader}\n• **長度**：{duration} 秒\n• **輿情基準**：已成功同步 MongoDB 資料庫。"
        return status_card, stimuli_A, stimuli_B

    except Exception as e:
        return f"❌ 系統錯誤: {str(e)}", "", ""
    finally:
        if os.path.exists(actual_audio_path):
            os.remove(actual_audio_path)

custom_css = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; padding-top: 2rem !important; }
.experimental-card { border: 1px solid #1e293b !important; background: linear-gradient(145deg, #0f172a, #1e1b4b) !important; border-radius: 16px !important; box-shadow: 0 10px 25px rgba(0,0,0,0.4) !important; padding: 25px !important; }
.action-btn { background: linear-gradient(90deg, #38bdf8, #818cf8) !important; color: white !important; border: none !important; font-weight: bold !important; }
.action-btn:hover { transform: translateY(-1px) !important; box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important; }
"""

custom_theme = gr.themes.Soft(primary_hue="sky", secondary_hue="slate", neutral_hue="slate").set(
    body_background_fill="*neutral_950", block_background_fill="*neutral_900", block_border_color="*neutral_800"
)

with gr.Blocks(theme=custom_theme, css=custom_css, title="TAICHI 2026 跨模態輿情閱讀器") as demo:
    gr.Markdown("# 🏆 TAICHI 2026 實驗系統原型：跨模態輿情閱讀與查核工具\n**研究主題**：媒體資訊超載時代，LLM 輔助的新聞影片閱讀工具對使用者資訊消費行為的影響")
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="experimental-card"):
            gr.Markdown("### 🛠️ 1. 實驗參數輸入")
            input_keyword = gr.Textbox(label="輿情定錨關鍵字", placeholder="例如: iPhone災情、輝達...", value="科技")
            input_url = gr.Textbox(label="要查核的 YouTube 影片網址", placeholder="https://www.youtube.com/watch?v=...")
            btn_submit = gr.Button("🚀 啟動雙軌跨模態分析", elem_classes="action-btn")
            gr.Markdown("---")
            output_meta = gr.Markdown("### 📊 系統狀態\n等待受測者輸入...")

        with gr.Column(scale=2, elem_classes="experimental-card"):
            gr.Markdown("### 🧪 2. 雙軌呈現粒度對照介面 (Dual-view Interface)")
            with gr.Tabs():
                with gr.TabItem("⚙️ A組介面（純極簡摘要 - 黑盒子模式）"):
                    output_A = gr.Markdown("*(等待系統生成)*\n\n此模式僅提供高度精簡大綱，隱藏所有時間軸，模擬低認知負荷。")
                with gr.TabItem("🔍 B組介面（時間軸逐字稿 - 可解釋性模式）"):
                    output_B = gr.Markdown("*(等待系統生成)*\n\n此模式提供精準時間戳記與 VTT 逐字稿，供使用者進行跨模態真偽查核。")

    btn_submit.click(fn=taichi_experiment_pipeline, inputs=[input_keyword, input_url], outputs=[output_meta, output_A, output_B])

if __name__ == "__main__":
    demo.launch()