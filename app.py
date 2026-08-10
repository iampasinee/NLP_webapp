import re
import random
import pandas as pd
import streamlit as st
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from pythainlp.util import normalize
from pythainlp.corpus.common import thai_stopwords

# ---------------------------------------------------------------------------
# 1) CONFIG & CSS STYLING (ปรับ UI ให้สวยงาม)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Review Analyzer AI", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
    .score-pos { color: #155724; font-weight: bold; font-size: 1.2em; }
    .score-neg { color: #721c24; font-weight: bold; font-size: 1.2em; }
    .score-neu { color: #383d41; font-weight: bold; font-size: 1.2em; }
    .badge-brand { background-color: #cce5ff; color: #004085; padding: 4px 10px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9em;}
    .badge-loc { background-color: #fff3cd; color: #856404; padding: 4px 10px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9em;}
    .badge-menu { background-color: #d1ecf1; color: #856404; padding: 4px 10px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9em;}
    .badge-pos { background-color: #d4edda; color: #155724; padding: 4px 10px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9em;}
    .badge-neg { background-color: #f8d7da; color: #721c24; padding: 4px 10px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9em;}
    .card { background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

STOPWORDS = set(thai_stopwords()) | {"ค่ะ", "ครับ", "นะ", "จ้า", "จ้ะ", "อ่ะ", "555", "5555"}

# รายการรีวิวตัวอย่าง สำหรับปุ่มสุ่ม
SAMPLE_REVIEWS_LIST = [
    "ร้านส้มตำเจ๊นางแซ่บมากกกก อร่อยสุดๆ ราคาไม่แพง พนักงานใจดี บริการรวดเร็ว อยู่แถวสยาม ลองสั่งต้มยำกุ้งด้วย เด็ดมาก!",
    "ไปกินหมูกระทะร้านนี้ที่ลาดพร้าวมา สกปรกมาก เนื้อก็เหนียว บริการช้า รอนานเกินไป ห่วยแตก ผิดหวังสุดๆ ไม่แนะนำเลย",
    "ร้านกาแฟเล็กๆ ใกล้อารีย์ บรรยากาศดีมาก กาแฟลาเต้หอมอร่อยดี แต่ราคาแพงไปนิดนึง โดยรวมโอเค พนักงานยิ้มแย้ม",
    "สั่งซูชิจากร้านชินคันเซ็นมาส่งที่บ้าน วัตถุดิบไม่สดเลย ข้าวแฉะ เค็มเกินไป รสชาติแย่มาก เสียดายเงิน",
    "ร้านชาบูตรงข้ามเซ็นทรัล น้ำซุปอร่อยดี หมูสไลด์บางนุ่ม บริการโอเค แต่แอร์ร้อนไปหน่อย แนะนำให้ไปช่วงเย็น"
]

# ---------------------------------------------------------------------------
# 2) WEIGHTED SENTIMENT DICTIONARY (ระบบคะแนนแบบมีน้ำหนัก)
# ---------------------------------------------------------------------------
# คำที่มีน้ำหนักมาก (+3, -3), น้ำหนักปานกลาง (+2, -2), และน้ำหนักน้อย (+1, -1)
SENTIMENT_WEIGHTS = {
    # เชิงบวก
    "อร่อยสุดๆ": 3, "แซ่บมาก": 3, "เด็ดมาก": 3, "สุดยอด": 3, "ดีเลิศ": 3, "ประทับใจสุดๆ": 3,
    "อร่อย": 2, "แซ่บ": 2, "คุ้มค่า": 2, "ประทับใจ": 2, "ดีมาก": 2, "หอม": 2, "สด": 2,
    "ดี": 1, "โอเค": 1, "พอใช้": 1, "ใช้ได้": 1, "สะอาด": 1, "น่ารัก": 1,
    
    # เชิงลบ
    "ห่วยแตก": -3, "แย่มาก": -3, "สกปรกมาก": -3, "ไม่แนะนำ": -3, "ผิดหวังสุดๆ": -3,
    "ไม่อร่อย": -2, "ผิดหวัง": -2, "สกปรก": -2, "แพงไป": -2, "รอนาน": -2, "เหนียว": -2, "แย่": -2,
    "ช้า": -1, "จืด": -1, "เค็ม": -1, "แพง": -1, "ร้อน": -1
}

# (ส่วน Dictionaries ของ Menu, Location, Brand คงเดิม)
MENU_DICT = ["ส้มตำ", "ต้มยำกุ้ง", "หมูกระทะ", "ชาบู", "กาแฟ", "ลาเต้", "ซูชิ", "ข้าวผัด", "ก๋วยเตี๋ยว", "เบอร์เกอร์"]
THAI_PROVINCES = ["กรุงเทพ", "เชียงใหม่", "ชลบุรี", "สยาม", "ทองหล่อ", "อารีย์", "อโศก", "เอกมัย", "ลาดพร้าว", "เซ็นทรัล"]

# ---------------------------------------------------------------------------
# 3) CORE FUNCTIONS
# ---------------------------------------------------------------------------
def clean_text(text: str):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text) # อร่อยยยยย -> อร่อยย
    return text.strip()

def extract_sentiment_weighted(text: str):
    """คำนวณ Sentiment แบบมีน้ำหนัก"""
    score = 0
    pos_words_found = []
    neg_words_found = []
    
    # วนลูปเช็คคำใน Dictionary
    for word, weight in SENTIMENT_WEIGHTS.items():
        if word in text:
            # เช็คคำปฏิเสธ (Negation) ง่ายๆ เช่น ถ้าเจอ "ไม่" นำหน้า
            if f"ไม่{word}" in text and weight > 0:
                score -= weight # กลับทางคะแนนเป็นลบ
                neg_words_found.append(f"ไม่{word}")
            else:
                score += weight
                if weight > 0:
                    pos_words_found.append(word)
                else:
                    neg_words_found.append(word)

    # ประเมินภาพรวมจาก Score
    if score >= 2:
        overall = "เชิงบวก (Positive) 🟢"
        css_class = "score-pos"
    elif score <= -2:
        overall = "เชิงลบ (Negative) 🔴"
        css_class = "score-neg"
    else:
        overall = "เป็นกลาง (Neutral) ⚪"
        css_class = "score-neu"
        
    return pos_words_found, neg_words_found, score, overall, css_class

def extract_entities(text: str):
    """ฟังก์ชันจำลอง NER สกัดทำเล และ เมนู"""
    found_menus = [m for m in MENU_DICT if m in text]
    found_locs = [l for l in THAI_PROVINCES if l in text]
    
    # Rule-based ดึงชื่อร้าน
    brand_match = re.search(r'(ร้าน|แบรนด์)([\w]+)', text)
    brand = brand_match.group(2) if brand_match else "ไม่ระบุ"
    
    return brand, found_locs, found_menus

# สร้าง HTML Badge สำหรับตกแต่ง
def make_badges(items, badge_class):
    if not items:
        return "-"
    return " ".join([f"<span class='{badge_class}'>{item}</span>" for item in items])

# ---------------------------------------------------------------------------
# 4) STREAMLIT UI
# ---------------------------------------------------------------------------
st.title("🍽️ Review Screening System")
st.markdown("ระบบวิเคราะห์รีวิวอัจฉริยะ (สกัดร้าน, ทำเล, เมนู และประเมินความรู้สึกแบบ Weight-Scoring)")

# การจัดการ Session State สำหรับปุ่มสุ่มรีวิว
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def set_random_review():
    st.session_state.user_input = random.choice(SAMPLE_REVIEWS_LIST)

# ปุ่มสุ่มรีวิว
st.button("🎲 สุ่มรีวิวตัวอย่าง (Random Review)", on_click=set_random_review, type="secondary")

# กล่องข้อความที่เชื่อมกับ Session State
user_text = st.text_area("ป้อนข้อความรีวิวสินค้าหรืออาหารที่นี่:", key="user_input", height=120)

if st.button("🔍 วิเคราะห์รีวิว (Analyze)", type="primary"):
    if not user_text.strip():
        st.warning("กรุณาป้อนข้อความรีวิวก่อนครับ")
    else:
        with st.spinner("กำลังประมวลผลด้วย NLP..."):
            cleaned_text = clean_text(user_text)
            brand, locs, menus = extract_entities(cleaned_text)
            pos_words, neg_words, score, overall_sentiment, score_css = extract_sentiment_weighted(cleaned_text)
            
            # วาด UI ผลลัพธ์
            st.markdown("---")
            st.subheader("📊 ผลการวิเคราะห์ข้อมูล (Analysis Results)")
            
            # ส่วนแสดง Score
            st.markdown(f"<div class='card'><h4>คะแนนความรู้สึก (Sentiment Score): <span class='{score_css}'>{score}</span> คะแนน</h4><p>แนวโน้ม: <b>{overall_sentiment}</b></p></div>", unsafe_allow_html=True)
            
            # ส่วนแสดง Entities แยกเป็นกล่อง 2 คอลัมน์
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📌 ข้อมูลที่สกัดได้ (Extracted Entities):**")
                st.markdown(f"**🏪 ชื่อร้าน/แบรนด์:** {make_badges([brand] if brand != 'ไม่ระบุ' else [], 'badge-brand')}", unsafe_allow_html=True)
                st.markdown(f"**📍 ทำเล/พิกัด:** {make_badges(locs, 'badge-loc')}", unsafe_allow_html=True)
                st.markdown(f"**🍜 เมนู:** {make_badges(menus, 'badge-menu')}", unsafe_allow_html=True)
                
            with col2:
                st.markdown("**💬 คำชม / คำติ (Keywords):**")
                st.markdown(f"**👍 เชิงบวก:** {make_badges(pos_words, 'badge-pos')}", unsafe_allow_html=True)
                st.markdown(f"**👎 เชิงลบ:** {make_badges(neg_words, 'badge-neg')}", unsafe_allow_html=True)
