# 📊 ITViec Employee Review Dashboard

> **Dashboard phân tích cảm xúc & phân cụm đánh giá công ty IT** dựa trên review ứng viên/nhân viên đăng trên ITViec.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<!-- Bạn nên thêm 1 ảnh chụp màn hình app (nếu có) để minh họa -->

---

## 🚀 Tính năng chính

### 🎯 Phân tích cảm xúc
- **Dự đoán cảm xúc**: Tích cực, trung tính, tiêu cực cho từng review bằng mô hình ML
- **Visualization**: WordCloud, BarChart với color-coding theo sentiment

### 🔍 Phân cụm đánh giá (Clustering)
- **Nhóm đặc trưng**: Xác định pattern review theo từng công ty
- **Đề xuất cải thiện**: Tự động generate recommendations dựa trên cluster analysis

### 📈 So sánh đa công ty
- **Dashboard comparative**: So sánh cảm xúc và tiêu chí giữa nhiều công ty
- **Interactive charts**: Bar charts, pie charts với hover effects

### 🕸️ Dashboard Radar
- **Multi-dimensional view**: Trực quan hóa 5 tiêu chí chính
  - 💰 Salary & benefits
  - 📚 Training & learning  
  - 🤝 Management care
  - 🎉 Culture & fun
  - 🏢 Office & workspace

### 🎨 UI/UX Features
- ✨ **Responsive design** với gradient styling
- 🔧 **AgGrid tables** với filter, sort, tooltip
- 🎯 **Advanced filtering** theo năm, vị trí, cảm xúc
- 📱 **Mobile-friendly** interface

---

## 📦 Cài đặt và chạy app

### 1️⃣ Yêu cầu hệ thống
- 🐍 **Python** >= 3.8 (best is 3.10)
- 🌐 **Browser**: Chrome/Edge/Firefox (UI tốt nhất trên desktop)

### 2️⃣ Clone source code
```bash
git clone <repo-url>
cd project1_app
```

### 3️⃣ Cài đặt dependencies
```bash
pip install -r requirements.txt
```

> **📋 Thư viện bao gồm**: streamlit, pandas, numpy, wordcloud, matplotlib, plotly, joblib, scikit-learn, openpyxl, streamlit-aggrid, xlrd

### 4️⃣ Chuẩn bị dữ liệu và model
Đặt các file sau vào thư mục `Du lieu cung cap/`:

```
📁 Du lieu cung cap/
├── 📄 Reviews.xlsx
├── 📄 Overview_Companies.xlsx
├── 📄 Overview_Reviews.xlsx
├── 🤖 sentiment_model.pkl
├── 🤖 cluster_model.pkl
└── 🤖 cluster_keywords.pkl
```

### 5️⃣ Chạy ứng dụng
```bash
streamlit run project1_app.py
```

🌐 **Truy cập**: http://localhost:8501

---

## 📊 Cấu trúc giao diện

| 📍 Vị trí | 🎛️ Chức năng | 📝 Mô tả |
|------------|---------------|-----------|
| **Sidebar** | Control Panel | Chọn công ty, chế độ phân tích, bộ lọc nâng cao |
| **Tab 1** | Tổng quan 1 công ty | Card summary, AgGrid table, WordCloud, đề xuất AI |
| **Tab 2** | So sánh nhiều công ty | Comparative charts, AgGrid với multi-company data |
| **Tab 3** | Phân cụm đánh giá | Cluster cards, keywords tags, pie chart distribution |
| **Tab 4** | Dashboard Radar | Interactive radar chart, metrics explanation |

---

## 🧠 Mô hình & Data Science

### 📈 Sentiment Analysis
- **Algorithm**: LinearSVC với TF-IDF vectorizer
- **Classes**: 3 classes (positive, neutral, negative)
- **Performance**: Pre-trained model với accuracy cao

### 🎯 Clustering Analysis  
- **Method**: KMeans/LDA clustering
- **Features**: Extracted keywords cho từng cluster
- **Output**: Tự động mapping tên cluster + improvement suggestions

### 🎨 Visualization
- **Interactive charts**: Plotly với hover effects
- **Color schemes**: Consistent blue theme
- **Responsive design**: Auto-adapt theo screen size

---

## 💡 Tùy biến & mở rộng

### 🎨 UI Customization
- ✅ Responsive layout (desktop + mobile)
- ✅ Blue gradient theme với hover effects
- ✅ Card-based design với smooth transitions

### 📊 Data Enhancement
- ✅ AgGrid với advanced features (filter, sort, search, tooltip)
- ✅ Dynamic filtering (năm, chức vụ, cảm xúc)
- ✅ Export capabilities (Excel, CSV)

### 🔮 Future Enhancements
- 📄 PDF report generation
- 🔄 Real-time data updates
- 🤖 Advanced ML models (BERT, GPT)
- 📱 Mobile app version

---

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📞 Liên hệ

👨‍💻 **Developer**: Lê Nguyễn Minh Quang & Nguyễn Quỳnh Oanh Thảo
📧 **Email**: minhquang030602t@example.com  
🌐 **Project**: ITViec Analysis Dashboard 2025


---

<div align="center">

**⭐ Nếu project hữu ích, hãy cho một star! ⭐**

Made with ❤️ by **Data Science & ML Team**

</div>