import streamlit as st
import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

# ตั้งค่าหน้าเว็บ (ช่วยเรื่องหน้าเว็บใช้งานง่าย)
st.set_page_config(page_title="Job Info Extractor", layout="centered")

st.title("💼 ระบบสกัดข้อมูลประกาศรับสมัครงาน")
st.write("แยกลักษณะงาน ชื่อบริษัท เงินเดือน ทักษะ และสถานที่ จากข้อความประกาศรับสมัครงาน")

# ฐานข้อมูลจำลอง (Dictionary/Keywords) สำหรับเทคนิค Rule-based ผสมผสาน
SKILL_KEYWORDS = ["python", "java", "sql", "react", "machine learning", "nlp", "data analysis", "excel", "c++", "aws", "docker"]
LOCATION_KEYWORDS = ["กรุงเทพ", "เชียงใหม่", "ขอนแก่น", "ภูเก็ต", "ชลบุรี", "ระยอง", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ", "wfh", "work from home"]

# รับข้อความจากผู้ใช้งาน
raw_text = st.text_area("วางข้อความประกาศรับสมัครงานที่นี่:", height=200)

if st.button("สกัดข้อมูล (Extract)"):
    if raw_text:
        with st.spinner("กำลังประมวลผล..."):
            
            # 1. Regex & Cleansing: ลบ Noise เช่น ลิงก์ URL
            clean_text = re.sub(r'http\S+|www.\S+', '', raw_text)
            
            # 2. Extract Salary (ใช้ Regex ดึงตัวเลขเงินเดือน เช่น 30,000 - 50,000 หรือ 30k)
            salary_pattern = r'(\d{2,3},?\d{3}\s*-\s*\d{2,3},?\d{3}|\d{2,3}[Kk]\s*-\s*\d{2,3}[Kk]|\d{2,3},?\d{3})'
            salaries = re.findall(salary_pattern, clean_text)
            extracted_salary = salaries[0] if salaries else "ไม่ระบุ"

            # 3. Tokenization & Normalization
            # ตัดคำและทำตัวพิมพ์เล็กทั้งหมดเพื่อเทียบ Keyword
            words = word_tokenize(clean_text.lower(), engine='newmm')
            stopwords = list(thai_stopwords())
            filtered_words = [w for w in words if w not in stopwords and w.strip() != '']

            # 4. Extract Skills (Keyword Matching จาก Tokenization)
            found_skills = list(set([skill for skill in SKILL_KEYWORDS if skill in clean_text.lower()]))
            
            # 5. Extract Location (Keyword Matching)
            found_locations = list(set([loc for loc in LOCATION_KEYWORDS if loc in clean_text.lower()]))
            
            # 6. Extract Company & Job Title (ใช้ Regex แบบง่าย โดยอิงจากบริบทคำนำหน้า)
            # หมายเหตุ: ในระบบจริงอาจต้องใช้ NER Model ที่เทรนมาเฉพาะ แต่สำหรับแบบทดสอบใช้ Rule-based ดึงคำหลัง Keyword ได้
            company_match = re.search(r'(บริษัท|company)[:\s]+([^\n\r]+)', clean_text, re.IGNORECASE)
            extracted_company = company_match.group(2).strip() if company_match else "ไม่ระบุ"
            
            position_match = re.search(r'(ตำแหน่ง|position|รับสมัคร)[:\s]+([^\n\r]+)', clean_text, re.IGNORECASE)
            extracted_position = position_match.group(2).strip() if position_match else "ไม่ระบุ"

            # แสดงผลลัพธ์
            st.success("สกัดข้อมูลสำเร็จ!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**📌 ตำแหน่งงาน:** {extracted_position}")
                st.markdown(f"**🏢 ชื่อบริษัท:** {extracted_company}")
                st.markdown(f"**💰 เงินเดือน:** {extracted_salary}")
            with col2:
                st.markdown(f"**📍 สถานที่:** {', '.join(found_locations) if found_locations else 'ไม่ระบุ'}")
                st.markdown(f"**🛠️ สกิลที่ต้องการ:** {', '.join(found_skills).title() if found_skills else 'ไม่ระบุ'}")
            
            st.divider()
            st.subheader("ข้อความที่ผ่านการ Cleansing แล้ว")
            st.info(clean_text)
    else:
        st.warning("กรุณาใส่ข้อความประกาศรับสมัครงานก่อนกดปุ่มครับ")
