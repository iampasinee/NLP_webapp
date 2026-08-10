"""
ระบบคัดกรองรีวิวสินค้า/อาหาร (Review Screening System) - อัปเกรด UI & Weighted Sentiment
สกัด: ชื่อแบรนด์/ร้าน, พิกัด/ทำเลร้าน, เมนู, คำชม/คำติ, คะแนนความรู้สึก
เทคนิคที่ใช้: Regex & Cleansing, Tokenization, Topic ID, POS Tagging, Weighted Sentiment
"""

import re
import io
import os
import random
from collections import Counter
import pandas as pd
import streamlit as st
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from pythainlp.util import normalize
from pythainlp.corpus.common import thai_stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansThai-Regular.ttf")

# ---------------------------------------------------------------------------
# 1) CONFIG, CSS STYLING & LEXICONS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ระบบคัดกรองรีวิวสินค้า/อาหาร", page_icon="🍽️", layout="wide")

# CSS สำหรับตกแต่ง UI ให้สวยงาม
st.markdown("""
<style>
    .score-pos { color: #155724; font-weight: bold; font-size: 1.5em; }
    .score-neg { color: #721c24; font-weight: bold; font-size: 1.5em; }
    .score-neu { color: #383d41; font-weight: bold; font-size: 1.5em; }
    .badge-brand { background-color: #cce5ff; color: #004085; padding: 5px 12px; border-radius: 20px; margin: 3px; display: inline-block; font-size: 0.9em; font-weight: 500;}
    .badge-loc { background-color: #fff3cd; color: #856404; padding: 5px 12px; border-radius: 20px; margin: 3px; display: inline-block; font-size: 0.9em; font-weight: 500;}
    .badge-menu { background-color: #d1ecf1; color: #0c5460; padding: 5px 12px; border-radius: 20px; margin: 3px; display: inline-block; font-size: 0.9em; font-weight: 500;}
    .badge-pos { background-color: #d4edda; color: #155724; padding: 5px 12px; border-radius: 20px; margin: 3px; display: inline-block; font-size: 0.9em; font-weight: 500;}
    .badge-neg { background-color: #f8d7da; color: #721c24; padding: 5px 12px; border-radius: 20px; margin: 3px; display: inline-block; font-size: 0.9em; font-weight: 500;}
    .badge-topic { background-color: #e2e3e5; color: #383d41; padding: 5px 12px; border-radius: 20px; margin: 3px; display: inline-block; font-size: 0.9em; font-weight: 500;}
    .result-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 5px solid #007bff;}
</style>
""", unsafe_allow_html=True)

# รายการรีวิวสุ่ม
SAMPLE_REVIEWS = [
    "ร้านส้มตำเจ๊นางแซ่บมากกกก อร่อยสุดๆ ราคาไม่แพง พนักงานใจดี บริการรวดเร็ว อยู่แถวสยาม ลองสั่งต้มยำกุ้งด้วย เด็ดมาก",
    "ร้าน Starbucks ทองหล่อ กาแฟลาเต้อร่อยดี แต่ราคาแพงไปนิด บรรยากาศดี สะอาด",
    "หมูกระทะร้านนี้ที่ลาดพร้าวไม่อร่อยเลย เนื้อแข็ง บริการช้ามาก รอนานเกินไป สกปรกมาก ไม่แนะนำ",
    "ไปกินก๋วยเตี๋ยวร้านเจ๊หน่อยแถวอโศก รสชาติจืดไปหน่อย แต่พนักงานใจดีมาก ราคาถูก คุ้มค่า",
    "The pizza at this place near Siam was amazing, great service and friendly staff, worth every baht!"
]

STOPWORDS = set(thai_stopwords()) | {"ค่ะ", "ครับ", "นะ", "จ้า", "จ้ะ", "อ่ะ", "555", "5555"} | {
    "near", "at", "in", "on", "the", "this", "that", "from", "was", "is", "a", "an"
}

BRAND_MARKERS = [r"ร้าน", r"แบรนด์", r"shop", r"brand", r"store"]
LOCATION_MARKERS = [r"แถว", r"ใกล้", r"ตรงข้าม", r"ซอย", r"ถนน", r"ที่", r"near", r"located at", r"located in"]
THAI_PROVINCES = [
    "กรุงเทพ", "กรุงเทพฯ", "เชียงใหม่", "เชียงราย", "ขอนแก่น", "นครราชสีมา", "โคราช",
    "ภูเก็ต", "ชลบุรี", "พัทยา", "สงขลา", "หาดใหญ่", "อยุธยา", "นนทบุรี", "ปทุมธานี",
    "สมุทรปราการ", "สยาม", "สีลม", "ทองหล่อ", "อารีย์", "อโศก", "เอกมัย", "ลาดพร้าว",
]

MENU_DICT = [
    "ส้มตำ", "ต้มยำกุ้ง", "ผัดไทย", "ข้าวผัด", "ก๋วยเตี๋ยว", "ข้าวมันไก่", "ข้าวซอย",
    "ลาบ", "น้ำตก", "แกงเขียวหวาน", "มัสมั่น", "ต้มข่าไก่", "หมูกระทะ", "ชาบู",
    "ปิ้งย่าง", "สุกี้", "ชานมไข่มุก", "กาแฟ", "ลาเต้", "เค้ก", "พิซซ่า", "เบอร์เกอร์",
    "สเต็ก", "พาสต้า", "สปาเก็ตตี้", "ซูชิ", "ราเมง", "ไก่ทอด", "หมูทอด", "ข้าวหมูแดง",
    "บะหมี่", "โจ๊ก", "ขนมปัง", "โดนัท", "ไอศกรีม", "yaki", "sushi", "ramen", "burger",
    "pizza", "steak", "coffee", "latte", "noodle", "fried rice",
]

# พจนานุกรมคำชม / คำติ (คงไว้เพื่อการค้นหาพื้นฐาน)
POSITIVE_WORDS = [
    "อร่อย", "แซ่บ", "ดี", "เยี่ยม", "สุดยอด", "ประทับใจ", "คุ้ม", "ถูก", "สด", "หอม",
    "น่ารัก", "ใจดี", "รวดเร็ว", "สะอาด", "แนะนำ", "ชอบ", "โดนใจ", "จัดเต็ม", "ฟิน",
    "good", "great", "delicious", "excellent", "amazing", "friendly", "fast", "clean",
    "recommend", "love", "tasty", "worth", "เด็ด",
]
NEGATIVE_WORDS = [
    "แย่", "ห่วย", "แพง", "ช้า", "ไม่อร่อย", "ผิดหวัง", "สกปรก", "เค็ม", "จืด", "หืน",
    "ไม่คุ้ม", "รอนาน", "หยาบคาย", "งอแง", "แข็ง", "ไหม้", "เหม็น", "ไม่พอใจ", "ไม่แนะนำ",
    "bad", "terrible", "expensive", "slow", "rude", "dirty", "disappointing", "cold",
    "overpriced", "awful", "not recommend",
]

# --- เพิ่มเติม: ระบบน้ำหนักคะแนน (Weighted Sentiment) ---
SENTIMENT_WEIGHTS = {
    # เชิงบวก
    "สุดยอด": 3, "อร่อยสุดๆ": 3, "เด็ดมาก": 3, "แซ่บมาก": 3, "ประทับใจสุดๆ": 3, "amazing": 3, "excellent": 3,
    "อร่อย": 2, "แซ่บ": 2, "ประทับใจ": 2, "คุ้มค่า": 2, "เด็ด": 2, "ดีมาก": 2, "หอม": 2, "สด": 2, "great": 2, "delicious": 2,
    "ดี": 1, "โอเค": 1, "คุ้ม": 1, "ถูก": 1, "น่ารัก": 1, "ใจดี": 1, "สะอาด": 1, "แนะนำ": 1, "ชอบ": 1, "ฟิน": 1, "good": 1,
    # เชิงลบ
    "ห่วยแตก": -3, "แย่มาก": -3, "สกปรกมาก": -3, "ผิดหวังสุดๆ": -3, "ไม่แนะนำ": -3, "terrible": -3, "awful": -3,
    "ไม่อร่อย": -2, "ผิดหวัง": -2, "สกปรก": -2, "แพงไป": -2, "รอนาน": -2, "หยาบคาย": -2, "แย่": -2, "ห่วย": -2,
    "ช้า": -1, "แพง": -1, "เค็ม": -1, "จืด": -1, "หืน": -1, "แข็ง": -1, "ไหม้": -1, "เหม็น": -1, "bad": -1, "slow": -1
}

TOPIC_KEYWORDS = {
    "รสชาติ/คุณภาพอาหาร": ["อร่อย", "รสชาติ", "เค็ม", "จืด", "หวาน", "สด", "หอม", "ไม่อร่อย", "taste", "delicious", "flavor", "แซ่บ", "เด็ด"],
    "บริการ": ["บริการ", "พนักงาน", "ใจดี", "หยาบคาย", "รอนาน", "รวดเร็ว", "service", "staff", "friendly", "rude", "ช้า"],
    "ราคา": ["ราคา", "แพง", "ถูก", "คุ้ม", "ไม่คุ้ม", "price", "expensive", "cheap", "worth"],
    "บรรยากาศ/ความสะอาด": ["บรรยากาศ", "สะอาด", "สกปรก", "ร้าน", "ตกแต่ง", "clean", "dirty", "ambience", "atmosphere"],
}

# ---------------------------------------------------------------------------
# 2) REGEX & CLEANSING
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE = re.compile(r"0\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]+", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(text: str):
    phones = PHONE_RE.findall(text)
    urls = URL_RE.findall(text)
    cleaned = URL_RE.sub(" ", text)
    cleaned = EMOJI_RE.sub(" ", cleaned)
    cleaned = REPEAT_CHAR_RE.sub(r"\1\1", cleaned)  
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()
    return cleaned, phones, urls

# ---------------------------------------------------------------------------
# 3) TOKENIZATION & NORMALIZATION
# ---------------------------------------------------------------------------
def tokenize_and_normalize(text: str):
    tokens = word_tokenize(text, engine="newmm")
    norm_tokens = [normalize(t) for t in tokens if t.strip() != ""]
    tokens_no_stop = [t for t in norm_tokens if t not in STOPWORDS and not t.isspace()]
    return norm_tokens, tokens_no_stop

def get_pos_tags(tokens):
    try:
        return pos_tag(tokens, engine="perceptron", corpus="orchid_ud")
    except Exception:
        return [(t, "NOUN") for t in tokens]

# ---------------------------------------------------------------------------
# 4) EXTRACTORS
# ---------------------------------------------------------------------------
def _next_meaningful_tokens(tokens, start_idx, max_words=2):
    picked = []
    i = start_idx + 1
    while i < len(tokens) and len(picked) < max_words:
        tok = tokens[i]
        if tok.isspace():
            i += 1
            continue
        if tok in STOPWORDS or PHONE_RE.match(tok) or tok.isdigit():
            break
        picked.append(tok)
        i += 1
    return "".join(picked) if picked else None

def extract_brand(text: str, tokens):
    found = set()
    thai_markers = {"ร้าน", "แบรนด์"}
    for idx, tok in enumerate(tokens):
        if tok in thai_markers:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=1)
            if candidate: found.add(candidate)
        if tok.lower() in {"shop", "brand", "store"}:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=2)
            if candidate: found.add(candidate)
    common_english_words = {"The", "This", "That", "It", "There", "They", "We", "You", "I"}
    for m in re.finditer(r"\b[A-Z][a-zA-Z]{2,}\b", text):
        word = m.group(0)
        if word not in common_english_words: found.add(word)
    return list(found)

def extract_location(text: str, tokens):
    found = set()
    for prov in THAI_PROVINCES:
        if prov in text: found.add(prov)
    thai_markers = {"แถว", "ใกล้", "ตรงข้าม", "ซอย", "ถนน", "ที่"}
    for idx, tok in enumerate(tokens):
        if tok in thai_markers:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=1)
            if candidate and candidate not in THAI_PROVINCES:
                found.add(candidate)
            elif candidate:
                found.add(candidate)
    for m in re.finditer(r"(?:near|located at|located in)\s+([A-Za-z]+)", text, flags=re.IGNORECASE):
        found.add(m.group(1).strip().rstrip(".,"))
    return list(found)

def extract_menu(text: str, tokens):
    found = set()
    for item in MENU_DICT:
        if item.lower() in text.lower(): found.add(item)
    verb_markers = {"สั่ง", "กิน", "ทาน"}
    for idx, tok in enumerate(tokens):
        if tok in verb_markers:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=2)
            if candidate: found.add(candidate)
    for m in re.finditer(r"(?:order|ordered)\s+([A-Za-z]+(?:\s[A-Za-z]+)?)", text, flags=re.IGNORECASE):
        found.add(m.group(1).strip().rstrip(".,"))
    return list(found)

NEGATION_WORDS = {"ไม่", "ไม่ค่อย", "ไม่ได้", "มิ", "ไม่สู้", "not", "no"}

def extract_sentiment_weighted(tokens, text: str):
    """ตรวจคำชม/คำติ พร้อมคำนวณคะแนน Weighted Scoring"""
    pos_hits, neg_hits = [], []
    total_score = 0
    
    lower_tokens = [t.lower() for t in tokens if t.strip() != ""]
    pos_set = {w.lower() for w in POSITIVE_WORDS}
    neg_set = {w.lower() for w in NEGATIVE_WORDS}

    def get_weight(word, is_positive):
        return SENTIMENT_WEIGHTS.get(word, 1 if is_positive else -1)

    for idx, tok in enumerate(lower_tokens):
        prev_tok = lower_tokens[idx - 1] if idx > 0 else ""
        negated = prev_tok in NEGATION_WORDS
        
        matched_pos = tok in pos_set or any(w in tok for w in pos_set if len(w) >= 2)
        matched_neg = tok in neg_set or any(w in tok for w in neg_set if len(w) >= 2)
        
        if matched_pos and not matched_neg:
            actual_word = tok if tok in pos_set else next((w for w in pos_set if w in tok), tok)
            weight = get_weight(actual_word, True)
            if negated:
                neg_hits.append(f"ไม่{tok}")
                total_score -= weight
            else:
                pos_hits.append(tok)
                total_score += weight
        elif matched_neg:
            actual_word = tok if tok in neg_set else next((w for w in neg_set if w in tok), tok)
            weight = get_weight(actual_word, False) # เป็นค่าลบ
            if negated:
                pos_hits.append(f"ไม่{tok}")
                total_score -= weight # ลบของลบเป็นบวก
            else:
                neg_hits.append(tok)
                total_score += weight

    low_text = text.lower()
    for w in NEGATIVE_WORDS:
        if " " in w and w in low_text:
            neg_hits.append(w)
            total_score += get_weight(w, False)

    pos_hits = list(dict.fromkeys(pos_hits))
    neg_hits = list(dict.fromkeys(neg_hits))

    if total_score > 0:
        overall = "เชิงบวก (Positive)"
        css_class = "score-pos"
    elif total_score < 0:
        overall = "เชิงลบ (Negative)"
        css_class = "score-neg"
    else:
        overall = "เป็นกลาง (Neutral)"
        css_class = "score-neu"
        
    return pos_hits, neg_hits, overall, total_score, css_class

def classify_topic(text: str):
    scores = {}
    low = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in low)
        if score > 0: scores[topic] = score
    if not scores: return ["ทั่วไป/อื่นๆ"]
    max_score = max(scores.values())
    return [t for t, s in scores.items() if s == max_score]

def analyze_review(text: str):
    cleaned, phones, urls = clean_text(text)
    all_tokens, tokens_no_stop = tokenize_and_normalize(cleaned)
    pos_tags = get_pos_tags(all_tokens)

    brand = extract_brand(cleaned, all_tokens)
    location = extract_location(cleaned, all_tokens)
    menu = extract_menu(cleaned, all_tokens)
    pos_hits, neg_hits, overall, score, _ = extract_sentiment_weighted(all_tokens, cleaned)
    topics = classify_topic(cleaned)

    return {
        "ข้อความต้นฉบับ": text,
        "ข้อความหลังทำความสะอาด": cleaned,
        "ชื่อแบรนด์/ร้าน": ", ".join(brand) if brand else "-",
        "ทำเล/พิกัด": ", ".join(location) if location else "-",
        "เมนู": ", ".join(menu) if menu else "-",
        "คำชม": ", ".join(pos_hits) if pos_hits else "-",
        "คำติ": ", ".join(neg_hits) if neg_hits else "-",
        "คะแนนความรู้สึก": score,
        "แนวโน้มความคิดเห็น": overall,
        "หมวดหมู่ (Topic)": ", ".join(topics),
        "เบอร์โทรที่พบ": ", ".join(phones) if phones else "-",
        "ลิงก์ที่พบ": ", ".join(urls) if urls else "-",
        "จำนวนคำ": len(tokens_no_stop),
        "_pos_tags": pos_tags,
    }

# ---------------------------------------------------------------------------
# 5) WORD CLOUD & BATCH HELPERS
# ---------------------------------------------------------------------------
def build_word_frequencies(texts, top_n=100):
    counter = Counter()
    for text in texts:
        cleaned, _, _ = clean_text(str(text))
        tokens = word_tokenize(cleaned, engine="newmm")
        for tok in tokens:
            tok = normalize(tok).strip()
            if not tok or tok in STOPWORDS or tok.isspace() or tok.isdigit() or not re.search(r"[ก-๙a-zA-Z]", tok) or len(tok) < 2:
                continue
            counter[tok] += 1
    return dict(counter.most_common(top_n))

def render_wordcloud(freqs):
    if not freqs: return None
    wc = WordCloud(font_path=FONT_PATH, width=900, height=450, background_color="white", colormap="viridis", max_words=80).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

def get_top_n_from_column(series, n=10):
    counter = Counter()
    for val in series:
        if not val or val == "-": continue
        for item in str(val).split(","):
            item = item.strip()
            if item and item != "-": counter[item] += 1
    if not counter: return pd.DataFrame(columns=["รายการ", "จำนวนครั้งที่พบ"])
    return pd.DataFrame(counter.most_common(n), columns=["รายการ", "จำนวนครั้งที่พบ"]).set_index("รายการ")

# HTML Badge Helper
def create_html_badges(items_str, badge_class):
    if not items_str or items_str == "-": return "-"
    items = [i.strip() for i in items_str.split(",")]
    return " ".join([f"<span class='{badge_class}'>{item}</span>" for item in items])


# ---------------------------------------------------------------------------
# 6) STREAMLIT UI
# ---------------------------------------------------------------------------
st.title("🍽️ ระบบวิเคราะห์และคัดกรองรีวิวอัจฉริยะ")
st.caption("สกัดชื่อแบรนด์ ทำเล เมนู และประเมินความรู้สึกแบบ Weight-Scoring (ไทย/อังกฤษ)")

with st.sidebar:
    st.header("ℹ️ เกี่ยวกับระบบ")
    st.markdown("""
**จุดเด่นที่อัปเกรด:**
- 🎲 **Random Review:** สุ่มรีวิวทดสอบได้ทันที
- ⚖️ **Weighted Scoring:** ประเมินอารมณ์แบบมีน้ำหนัก (เช่น แซ่บมาก +3, โอเค +1)
- 🎨 **Modern UI:** แสดงผลด้วย Badge สวยงามอ่านง่าย
    """)
    st.divider()
    st.caption("พัฒนาโดยใช้ Streamlit + PyThaiNLP")

tab1, tab2 = st.tabs(["📝 วิเคราะห์รีวิวเดี่ยว (Single)", "📂 วิเคราะห์แบบไฟล์ (Batch)"])

# ---- TAB 1: Single Review ----
with tab1:
    st.subheader("ป้อนข้อความรีวิว")
    
    # Session State สำหรับปุ่มสุ่ม
    if "user_input" not in st.session_state:
        st.session_state.user_input = SAMPLE_REVIEWS[0]

    def set_random_review():
        st.session_state.user_input = random.choice(SAMPLE_REVIEWS)

    st.button("🎲 สุ่มรีวิวตัวอย่าง (Random Review)", on_click=set_random_review, type="secondary")
    
    user_text = st.text_area("ข้อความรีวิว (ภาษาไทยหรืออังกฤษ)", key="user_input", height=120)

    if st.button("🔍 วิเคราะห์ข้อมูล (Analyze)", type="primary"):
        if not user_text.strip():
            st.warning("กรุณาป้อนข้อความรีวิวก่อน")
        else:
            result = analyze_review(user_text)
            pos_tags = result.pop("_pos_tags")
            
            _, _, overall, score, css_class = extract_sentiment_weighted(word_tokenize(result["ข้อความหลังทำความสะอาด"], engine="newmm"), result["ข้อความหลังทำความสะอาด"])

            # แสดงผลแบบสวยงามด้วย HTML/CSS
            st.markdown(f"""
            <div class='result-card'>
                <h4 style='margin-top:0;'>📊 คะแนนความรู้สึก (Sentiment Score): <span class='{css_class}'>{score:+}</span></h4>
                <p style='margin-bottom:0;'>แนวโน้มความคิดเห็นโดยรวม: <b>{overall}</b></p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📌 เอนทิตีที่สกัดได้ (Extracted Entities)")
                st.markdown(f"**🏪 แบรนด์/ร้าน:** {create_html_badges(result['ชื่อแบรนด์/ร้าน'], 'badge-brand')}", unsafe_allow_html=True)
                st.markdown(f"**📍 ทำเล/พิกัด:** {create_html_badges(result['ทำเล/พิกัด'], 'badge-loc')}", unsafe_allow_html=True)
                st.markdown(f"**🍜 เมนู:** {create_html_badges(result['เมนู'], 'badge-menu')}", unsafe_allow_html=True)
                st.markdown(f"**🏷️ หมวดหมู่:** {create_html_badges(result['หมวดหมู่ (Topic)'], 'badge-topic')}", unsafe_allow_html=True)
            with col2:
                st.markdown("##### 💬 คำสำคัญ (Keywords)")
                st.markdown(f"**👍 คำชม:** {create_html_badges(result['คำชม'], 'badge-pos')}", unsafe_allow_html=True)
                st.markdown(f"**👎 คำติ:** {create_html_badges(result['คำติ'], 'badge-neg')}", unsafe_allow_html=True)
                st.markdown(f"**📞 ติดต่อ:** {result['เบอร์โทรที่พบ']} | **🔗 ลิงก์:** {result['ลิงก์ที่พบ']}")

            with st.expander("🔬 ดูรายละเอียดข้อมูลดิบและ POS Tags"):
                st.table(pd.DataFrame({"รายการ": list(result.keys()), "ผลลัพธ์": list(result.values())}))
                st.write("**POS Tags:**", pos_tags)

# ---- TAB 2: Batch via CSV ----
with tab2:
    st.subheader("อัปโหลดไฟล์รีวิว")
    st.caption("รองรับไฟล์ .txt (1 บรรทัด = 1 รีวิว) หรือ .csv ที่มีคอลัมน์ `review`")

    uploaded = st.file_uploader("เลือกไฟล์ CSV หรือ TXT", type=["csv", "txt"])
    use_sample = st.checkbox("ใช้ไฟล์ตัวอย่าง sample_reviews.txt แทน", value=False)

    def load_txt_reviews(file_obj_or_path):
        if hasattr(file_obj_or_path, "read"):
            raw = file_obj_or_path.read()
            content = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        else:
            with open(file_obj_or_path, "r", encoding="utf-8") as f: content = f.read()
        lines = [line.strip() for line in content.splitlines()]
        return pd.DataFrame({"review": [line for line in lines if line]})

    df_input = None
    if uploaded is not None:
        df_input = load_txt_reviews(uploaded) if uploaded.name.lower().endswith(".txt") else pd.read_csv(uploaded)
    elif use_sample:
        try: df_input = load_txt_reviews("sample_reviews.txt")
        except FileNotFoundError: st.error("ไม่พบไฟล์ sample_reviews.txt ในโปรเจกต์")

    if df_input is not None:
        if "review" not in df_input.columns:
            st.error("ไม่พบคอลัมน์ 'review' ในไฟล์")
        else:
            st.success(f"โหลดข้อมูลสำเร็จ: {len(df_input)} รีวิว")
            if st.button("🚀 ประมวลผลทั้งหมด", type="primary"):
                results = []
                progress = st.progress(0, text="กำลังประมวลผล...")
                for i, row in df_input.iterrows():
                    r = analyze_review(str(row["review"]))
                    r.pop("_pos_tags", None)
                    results.append(r)
                    progress.progress((i + 1) / len(df_input), text=f"ประมวลผล {i+1}/{len(df_input)}")
                progress.empty()

                result_df = pd.DataFrame(results)
                st.markdown("### ผลลัพธ์ทั้งหมด")
                st.dataframe(result_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### สัดส่วนแนวโน้มความคิดเห็น")
                    st.bar_chart(result_df["แนวโน้มความคิดเห็น"].value_counts())
                with col2:
                    st.markdown("#### สัดส่วนหมวดหมู่ (Topic)")
                    st.bar_chart(result_df["หมวดหมู่ (Topic)"].value_counts())

                st.divider()
                st.markdown("### ☁️ Word Cloud คำที่พบบ่อยในรีวิวทั้งหมด")
                freqs = build_word_frequencies(df_input["review"].tolist())
                fig = render_wordcloud(freqs)
                if fig: st.pyplot(fig, use_container_width=True)

                st.divider()
                st.markdown("### 🏆 Top-N ร้าน/แบรนด์ และเมนูที่ถูกพูดถึงมากที่สุด")
                top_n = st.slider("จำนวนอันดับที่ต้องการแสดง (Top-N)", 3, 20, 10)
                col3, col4 = st.columns(2)
                with col3:
                    top_brand_df = get_top_n_from_column(result_df["ชื่อแบรนด์/ร้าน"], n=top_n)
                    if not top_brand_df.empty: st.bar_chart(top_brand_df)
                with col4:
                    top_menu_df = get_top_n_from_column(result_df["เมนู"], n=top_n)
                    if not top_menu_df.empty: st.bar_chart(top_menu_df)

                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                st.download_button("⬇️ ดาวน์โหลดผลลัพธ์เป็น CSV", data=csv_buffer.getvalue(), file_name="review_analysis_results.csv", mime="text/csv")
