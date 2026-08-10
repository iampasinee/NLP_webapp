"""
ระบบคัดกรองรีวิวสินค้า/อาหาร (Review Screening System)
สกัด: ชื่อแบรนด์/ร้าน, พิกัด/ทำเลร้าน, เมนู, คำชม/คำติ
เทคนิคที่ใช้: Regex & Cleansing, Tokenization & Normalization, Topic Identification, POS Tagging (NER-lite)
"""

import re
import io
import os
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
# 1) CONFIG / LEXICONS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ระบบคัดกรองรีวิวสินค้า/อาหาร", page_icon="🍽️", layout="wide")

STOPWORDS = set(thai_stopwords()) | {"ค่ะ", "ครับ", "นะ", "จ้า", "จ้ะ", "อ่ะ", "555", "5555"} | {
    "near", "at", "in", "on", "the", "this", "that", "from", "was", "is", "a", "an"
}

# คำบ่งชี้ร้าน/แบรนด์ (ไทย/อังกฤษ)
BRAND_MARKERS = [r"ร้าน", r"แบรนด์", r"shop", r"brand", r"store"]

# คำบ่งชี้ทำเล/พิกัด
LOCATION_MARKERS = [r"แถว", r"ใกล้", r"ตรงข้าม", r"ซอย", r"ถนน", r"ที่", r"near", r"located at", r"located in"]
THAI_PROVINCES = [
    "กรุงเทพ", "กรุงเทพฯ", "เชียงใหม่", "เชียงราย", "ขอนแก่น", "นครราชสีมา", "โคราช",
    "ภูเก็ต", "ชลบุรี", "พัทยา", "สงขลา", "หาดใหญ่", "อยุธยา", "นนทบุรี", "ปทุมธานี",
    "สมุทรปราการ", "สยาม", "สีลม", "ทองหล่อ", "อารีย์", "อโศก", "เอกมัย", "ลาดพร้าว",
]

# พจนานุกรมเมนู/อาหารยอดนิยม (ขยายเพิ่มได้)
MENU_DICT = [
    "ส้มตำ", "ต้มยำกุ้ง", "ผัดไทย", "ข้าวผัด", "ก๋วยเตี๋ยว", "ข้าวมันไก่", "ข้าวซอย",
    "ลาบ", "น้ำตก", "แกงเขียวหวาน", "มัสมั่น", "ต้มข่าไก่", "หมูกระทะ", "ชาบู",
    "ปิ้งย่าง", "สุกี้", "ชานมไข่มุก", "กาแฟ", "ลาเต้", "เค้ก", "พิซซ่า", "เบอร์เกอร์",
    "สเต็ก", "พาสต้า", "สปาเก็ตตี้", "ซูชิ", "ราเมง", "ไก่ทอด", "หมูทอด", "ข้าวหมูแดง",
    "บะหมี่", "โจ๊ก", "ขนมปัง", "โดนัท", "ไอศกรีม", "yaki", "sushi", "ramen", "burger",
    "pizza", "steak", "coffee", "latte", "noodle", "fried rice",
]

# พจนานุกรมคำชม / คำติ (ไทย/อังกฤษ)
POSITIVE_WORDS = [
    "อร่อย", "แซ่บ", "ดี", "เยี่ยม", "สุดยอด", "ประทับใจ", "คุ้ม", "ถูก", "สด", "หอม",
    "น่ารัก", "ใจดี", "รวดเร็ว", "สะอาด", "แนะนำ", "ชอบ", "โดนใจ", "จัดเต็ม", "ฟิน",
    "good", "great", "delicious", "excellent", "amazing", "friendly", "fast", "clean",
    "recommend", "love", "tasty", "worth",
]
NEGATIVE_WORDS = [
    "แย่", "ห่วย", "แพง", "ช้า", "ไม่อร่อย", "ผิดหวัง", "สกปรก", "เค็ม", "จืด", "หืน",
    "ไม่คุ้ม", "รอนาน", "หยาบคาย", "งอแง", "แข็ง", "ไหม้", "เหม็น", "ไม่พอใจ", "ไม่แนะนำ",
    "bad", "terrible", "expensive", "slow", "rude", "dirty", "disappointing", "cold",
    "overpriced", "awful", "not recommend",
]

TOPIC_KEYWORDS = {
    "รสชาติ/คุณภาพอาหาร": ["อร่อย", "รสชาติ", "เค็ม", "จืด", "หวาน", "สด", "หอม", "ไม่อร่อย", "taste", "delicious", "flavor"],
    "บริการ": ["บริการ", "พนักงาน", "ใจดี", "หยาบคาย", "รอนาน", "รวดเร็ว", "service", "staff", "friendly", "rude"],
    "ราคา": ["ราคา", "แพง", "ถูก", "คุ้ม", "ไม่คุ้ม", "price", "expensive", "cheap", "worth"],
    "บรรยากาศ/ความสะอาด": ["บรรยากาศ", "สะอาด", "สกปรก", "ร้าน", "ตกแต่ง", "clean", "dirty", "ambience", "atmosphere"],
}

# ---------------------------------------------------------------------------
# 2) REGEX & CLEANSING
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE = re.compile(r"0\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # อร่อยยยยย -> อร่อยย
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]+", flags=re.UNICODE
)
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str):
    """Regex & Cleansing: ลบ noise/emoji/ลิงก์ แต่ดึงเบอร์โทรและลิงก์เก็บไว้ต่างหากก่อนลบ"""
    phones = PHONE_RE.findall(text)
    urls = URL_RE.findall(text)

    cleaned = URL_RE.sub(" ", text)
    cleaned = EMOJI_RE.sub(" ", cleaned)
    cleaned = REPEAT_CHAR_RE.sub(r"\1\1", cleaned)  # ตัดอักษรซ้ำเกิน 2 ตัว
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


# ---------------------------------------------------------------------------
# 4) POS TAGGING (ใช้แทน NER แบบ rule-based/lightweight)
# ---------------------------------------------------------------------------
def get_pos_tags(tokens):
    try:
        return pos_tag(tokens, engine="perceptron", corpus="orchid_ud")
    except Exception:
        return [(t, "NOUN") for t in tokens]


# ---------------------------------------------------------------------------
# 5) EXTRACTORS: brand / location / menu / sentiment / topic
# ---------------------------------------------------------------------------
def _next_meaningful_tokens(tokens, start_idx, max_words=2):
    """เก็บ token ถัดไปที่ไม่ใช่ช่องว่าง/stopword/ตัวเลข จนครบ max_words คำ"""
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
    """สกัดชื่อแบรนด์/ร้าน: ใช้ token-based matching (ภาษาไทยไม่มีช่องว่างระหว่างคำ
    จึงต้อง tokenize ก่อนแล้วดูคำถัดจาก marker แทนการใช้ regex กับ raw string ตรงๆ)"""
    found = set()
    thai_markers = {"ร้าน", "แบรนด์"}
    for idx, tok in enumerate(tokens):
        if tok in thai_markers:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=1)
            if candidate:
                found.add(candidate)
        if tok.lower() in {"shop", "brand", "store"}:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=2)
            if candidate:
                found.add(candidate)
    # เดา proper noun ภาษาอังกฤษที่ขึ้นต้นตัวใหญ่ (เช่น McDonald, Starbucks)
    common_english_words = {"The", "This", "That", "It", "There", "They", "We", "You", "I"}
    for m in re.finditer(r"\b[A-Z][a-zA-Z]{2,}\b", text):
        word = m.group(0)
        if word not in common_english_words:
            found.add(word)
    return list(found)


def extract_location(text: str, tokens):
    found = set()
    for prov in THAI_PROVINCES:
        if prov in text:
            found.add(prov)
    thai_markers = {"แถว", "ใกล้", "ตรงข้าม", "ซอย", "ถนน", "ที่"}
    for idx, tok in enumerate(tokens):
        if tok in thai_markers:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=1)
            if candidate and candidate not in THAI_PROVINCES:  # หลีกเลี่ยงนับซ้ำ
                found.add(candidate)
            elif candidate:
                found.add(candidate)
    # English "near <place>" / "located at/in <place>" (จำกัด 1 คำ กันจับคำกริยาต่อท้ายมาด้วย)
    for m in re.finditer(r"(?:near|located at|located in)\s+([A-Za-z]+)", text, flags=re.IGNORECASE):
        found.add(m.group(1).strip().rstrip(".,"))
    return list(found)


def extract_menu(text: str, tokens):
    found = set()
    for item in MENU_DICT:
        if item.lower() in text.lower():
            found.add(item)
    # เสริมด้วย pattern กิน/สั่ง/ทาน + คำถัดไป (token-based)
    verb_markers = {"สั่ง", "กิน", "ทาน"}
    for idx, tok in enumerate(tokens):
        if tok in verb_markers:
            candidate = _next_meaningful_tokens(tokens, idx, max_words=2)
            if candidate:
                found.add(candidate)
    for m in re.finditer(r"(?:order|ordered)\s+([A-Za-z]+(?:\s[A-Za-z]+)?)", text, flags=re.IGNORECASE):
        found.add(m.group(1).strip().rstrip(".,"))
    return list(found)


NEGATION_WORDS = {"ไม่", "ไม่ค่อย", "ไม่ได้", "มิ", "ไม่สู้", "not", "no"}


def extract_sentiment(tokens, text: str):
    """ตรวจคำชม/คำติ แบบ token-based พร้อมจัดการคำปฏิเสธ (negation) เบื้องต้น
    เช่น 'ไม่อร่อย' ต้องไม่ถูกนับเป็นคำชม 'อร่อย'"""
    pos_hits, neg_hits = [], []
    lower_tokens = [t.lower() for t in tokens if t.strip() != ""]  # ตัด token ที่เป็นช่องว่างออก แต่ยังคง stopword (เช่น 'ไม่') ไว้เพื่อเช็ก negation
    pos_set = {w.lower() for w in POSITIVE_WORDS}
    neg_set = {w.lower() for w in NEGATIVE_WORDS}

    for idx, tok in enumerate(lower_tokens):
        prev_tok = lower_tokens[idx - 1] if idx > 0 else ""
        negated = prev_tok in NEGATION_WORDS
        # จับคำที่ตรงเป๊ะ หรือเป็นคำผสมที่มีคำในดิกชันนารีอยู่ (เช่น 'ราคาแพง' มี 'แพง' อยู่ข้างใน)
        matched_pos = tok in pos_set or any(w in tok for w in pos_set if len(w) >= 2)
        matched_neg = tok in neg_set or any(w in tok for w in neg_set if len(w) >= 2)
        if matched_pos and not matched_neg:
            (neg_hits if negated else pos_hits).append(tok)
        elif matched_neg:
            (pos_hits if negated else neg_hits).append(tok)

    # เผื่อกรณีวลีภาษาอังกฤษที่มีเว้นวรรค เช่น "not recommend"
    low_text = text.lower()
    for w in NEGATIVE_WORDS:
        if " " in w and w in low_text:
            neg_hits.append(w)

    pos_hits = list(dict.fromkeys(pos_hits))
    neg_hits = list(dict.fromkeys(neg_hits))

    if len(pos_hits) > len(neg_hits):
        overall = "เชิงบวก (Positive)"
    elif len(neg_hits) > len(pos_hits):
        overall = "เชิงลบ (Negative)"
    else:
        overall = "เป็นกลาง (Neutral)"
    return pos_hits, neg_hits, overall


def classify_topic(text: str):
    scores = {}
    low = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in low)
        if score > 0:
            scores[topic] = score
    if not scores:
        return ["ทั่วไป/อื่นๆ"]
    max_score = max(scores.values())
    return [t for t, s in scores.items() if s == max_score]


def analyze_review(text: str):
    cleaned, phones, urls = clean_text(text)
    all_tokens, tokens_no_stop = tokenize_and_normalize(cleaned)
    pos_tags = get_pos_tags(all_tokens)

    brand = extract_brand(cleaned, all_tokens)
    location = extract_location(cleaned, all_tokens)
    menu = extract_menu(cleaned, all_tokens)
    pos_hits, neg_hits, overall = extract_sentiment(all_tokens, cleaned)
    topics = classify_topic(cleaned)

    return {
        "ข้อความต้นฉบับ": text,
        "ข้อความหลังทำความสะอาด": cleaned,
        "ชื่อแบรนด์/ร้าน": ", ".join(brand) if brand else "-",
        "ทำเล/พิกัด": ", ".join(location) if location else "-",
        "เมนู": ", ".join(menu) if menu else "-",
        "คำชม": ", ".join(pos_hits) if pos_hits else "-",
        "คำติ": ", ".join(neg_hits) if neg_hits else "-",
        "แนวโน้มความคิดเห็น": overall,
        "หมวดหมู่ (Topic)": ", ".join(topics),
        "เบอร์โทรที่พบ": ", ".join(phones) if phones else "-",
        "ลิงก์ที่พบ": ", ".join(urls) if urls else "-",
        "จำนวนคำ (หลังตัด stopword)": len(tokens_no_stop),
        "_pos_tags": pos_tags,
    }


def build_word_frequencies(texts, top_n=100):
    """รวม token จากรีวิวทั้งหมด นับความถี่ (ตัด stopword/เครื่องหมาย/ตัวเลขออก) สำหรับทำ Word Cloud"""
    counter = Counter()
    for text in texts:
        cleaned, _, _ = clean_text(str(text))
        tokens = word_tokenize(cleaned, engine="newmm")
        for tok in tokens:
            tok = normalize(tok).strip()
            if not tok or tok in STOPWORDS or tok.isspace():
                continue
            if tok.isdigit() or not re.search(r"[ก-๙a-zA-Z]", tok):
                continue
            if len(tok) < 2:
                continue
            counter[tok] += 1
    return dict(counter.most_common(top_n))


def render_wordcloud(freqs):
    if not freqs:
        return None
    wc = WordCloud(
        font_path=FONT_PATH,
        width=900,
        height=450,
        background_color="white",
        colormap="viridis",
        max_words=80,
    ).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def get_top_n_from_column(series, n=10):
    """นับความถี่จากคอลัมน์ที่เป็น comma-separated string (เช่น 'ส้มตำ, ต้มยำกุ้ง') คืนค่าเป็น DataFrame สำหรับ bar chart"""
    counter = Counter()
    for val in series:
        if not val or val == "-":
            continue
        for item in str(val).split(","):
            item = item.strip()
            if item and item != "-":
                counter[item] += 1
    if not counter:
        return pd.DataFrame(columns=["รายการ", "จำนวนครั้งที่พบ"])
    top = counter.most_common(n)
    return pd.DataFrame(top, columns=["รายการ", "จำนวนครั้งที่พบ"]).set_index("รายการ")


# ---------------------------------------------------------------------------
# 6) STREAMLIT UI
# ---------------------------------------------------------------------------
st.title("🍽️ ระบบคัดกรองรีวิวสินค้า/อาหาร")
st.caption("ประมวลผลข้อความรีวิว (ไทย/อังกฤษ) เพื่อสกัดชื่อแบรนด์/ร้าน, ทำเล/พิกัด, เมนู และคำชม/คำติ")

with st.sidebar:
    st.header("ℹ️ เกี่ยวกับระบบ")
    st.markdown(
        """
**Domain:** รีวิวสินค้า/อาหาร

**สิ่งที่ระบบสกัด:**
- ชื่อแบรนด์/ร้าน
- ทำเล/พิกัดร้าน
- เมนูอาหาร/เครื่องดื่ม
- คำชม / คำติ + แนวโน้มความคิดเห็นโดยรวม
- หมวดหมู่ของรีวิว (รสชาติ, บริการ, ราคา, บรรยากาศ)

**การวิเคราะห์แบบ batch ยังมี:**
- Word Cloud คำที่พบบ่อยทั้งชุดข้อมูล
- Top-N ร้าน/แบรนด์ และเมนูยอดนิยม

**เทคนิค NLP ที่ใช้:**
- Regex & Cleansing
- Tokenization & Normalization (PyThaiNLP)
- Topic Identification (keyword-based)
- POS Tagging (ใช้ระบุคำนาม/คุณศัพท์ประกอบการสกัด)
        """
    )
    st.divider()
    st.caption("พัฒนาโดยใช้ Streamlit + PyThaiNLP")

tab1, tab2 = st.tabs(["📝 วิเคราะห์รีวิวเดี่ยว", "📂 วิเคราะห์แบบไฟล์ (batch)"])

# ---- TAB 1: single review ----
with tab1:
    st.subheader("ป้อนข้อความรีวิว")
    default_text = "ร้านส้มตำแซ่บมากกกก อร่อยสุดๆ ราคาไม่แพง พนักงานใจดี บริการรวดเร็ว อยู่แถวสยาม โทร 0812345678 ลองสั่งต้มยำกุ้งด้วย เด็ดมาก"
    user_text = st.text_area("ข้อความรีวิว (ภาษาไทยหรืออังกฤษ)", value=default_text, height=120)

    if st.button("🔍 วิเคราะห์รีวิว", type="primary"):
        if not user_text.strip():
            st.warning("กรุณาป้อนข้อความรีวิวก่อน")
        else:
            result = analyze_review(user_text)
            pos_tags = result.pop("_pos_tags")

            col1, col2, col3 = st.columns(3)
            col1.metric("แนวโน้มความคิดเห็น", result["แนวโน้มความคิดเห็น"])
            col2.metric("ชื่อแบรนด์/ร้าน", result["ชื่อแบรนด์/ร้าน"])
            col3.metric("หมวดหมู่", result["หมวดหมู่ (Topic)"])

            st.markdown("### ผลการสกัดข้อมูล")
            display_result = {k: v for k, v in result.items() if k != "ข้อความต้นฉบับ"}
            st.table(pd.DataFrame(display_result.items(), columns=["รายการ", "ผลลัพธ์"]))

            with st.expander("🔬 รายละเอียดเชิงเทคนิค (Tokenization & POS Tagging)"):
                st.write("**Token ทั้งหมด:**", [t for t in word_tokenize(result['ข้อความหลังทำความสะอาด'], engine='newmm') if t.strip()])
                st.write("**POS Tags:**", pos_tags)

# ---- TAB 2: batch via CSV ----
with tab2:
    st.subheader("อัปโหลดไฟล์รีวิว")
    st.caption(
        "รองรับ 2 รูปแบบ: (1) ไฟล์ **.txt** โดยให้ 1 บรรทัด = 1 รีวิว (บรรทัดว่างใช้คั่นได้) "
        "หรือ (2) ไฟล์ **.csv** ที่มีคอลัมน์ชื่อ `review`"
    )

    uploaded = st.file_uploader("เลือกไฟล์ CSV หรือ TXT", type=["csv", "txt"])

    use_sample = st.checkbox("ใช้ไฟล์ตัวอย่าง sample_reviews.txt แทน", value=False)

    def load_txt_reviews(file_obj_or_path):
        """อ่านไฟล์ .txt โดยถือว่า 1 บรรทัดที่ไม่ว่าง = 1 รีวิว (บรรทัดว่างใช้คั่นเพื่อความอ่านง่าย จะถูกข้าม)"""
        if hasattr(file_obj_or_path, "read"):
            raw = file_obj_or_path.read()
            content = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        else:
            with open(file_obj_or_path, "r", encoding="utf-8") as f:
                content = f.read()
        lines = [line.strip() for line in content.splitlines()]
        reviews = [line for line in lines if line]  # ตัดบรรทัดว่างทิ้ง
        return pd.DataFrame({"review": reviews})

    df_input = None
    if uploaded is not None:
        if uploaded.name.lower().endswith(".txt"):
            df_input = load_txt_reviews(uploaded)
        else:
            df_input = pd.read_csv(uploaded)
    elif use_sample:
        try:
            df_input = load_txt_reviews("sample_reviews.txt")
        except FileNotFoundError:
            st.error("ไม่พบไฟล์ sample_reviews.txt ในโปรเจกต์")

    if df_input is not None:
        if "review" not in df_input.columns:
            st.error("ไม่พบคอลัมน์ 'review' ในไฟล์ที่อัปโหลด")
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
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("ไม่มีคำที่นับได้เพียงพอสำหรับสร้าง Word Cloud")

                st.divider()
                st.markdown("### 🏆 Top-N ร้าน/แบรนด์ และเมนูที่ถูกพูดถึงมากที่สุด")
                top_n = st.slider("จำนวนอันดับที่ต้องการแสดง (Top-N)", min_value=3, max_value=20, value=10)
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("#### ร้าน/แบรนด์ยอดนิยม")
                    top_brand_df = get_top_n_from_column(result_df["ชื่อแบรนด์/ร้าน"], n=top_n)
                    if not top_brand_df.empty:
                        st.bar_chart(top_brand_df)
                    else:
                        st.info("ไม่พบชื่อร้าน/แบรนด์ในรีวิวชุดนี้")
                with col4:
                    st.markdown("#### เมนูยอดนิยม")
                    top_menu_df = get_top_n_from_column(result_df["เมนู"], n=top_n)
                    if not top_menu_df.empty:
                        st.bar_chart(top_menu_df)
                    else:
                        st.info("ไม่พบเมนูในรีวิวชุดนี้")

                st.divider()
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                st.download_button(
                    "⬇️ ดาวน์โหลดผลลัพธ์เป็น CSV",
                    data=csv_buffer.getvalue(),
                    file_name="review_analysis_results.csv",
                    mime="text/csv",
                )
