import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go

# --- LOAD DATA ---
@st.cache_data
def load_data():
    reviews = pd.read_excel("Du lieu cung cap/Reviews.xlsx")
    overview_companies = pd.read_excel("Du lieu cung cap/Overview_Companies.xlsx")
    overview_reviews = pd.read_excel("Du lieu cung cap/Overview_Reviews.xlsx")
    return reviews, overview_companies, overview_reviews

reviews, overview_companies, overview_reviews = load_data()

sentiment_model = joblib.load("sentiment_model.pkl")
cluster_model = joblib.load("cluster_model.pkl")
cluster_keywords = joblib.load("cluster_keywords.pkl")

# --- ĐỀ XUẤT CẢI THIỆN THEO CỤM ---
cluster_improvement = {
    0: "Tăng cường training, lộ trình phát triển cá nhân rõ ràng.",
    1: "Cải thiện phúc lợi, xem xét tăng lương và chế độ đãi ngộ.",
    2: "Xây dựng văn hóa doanh nghiệp gắn kết, đa dạng hoạt động team-building.",
    3: "Tăng cường truyền thông nội bộ, chú trọng feedback nhân viên.",
}

# --- SIDEBAR LỰA CHỌN ---
st.sidebar.title("Phân tích dữ liệu ITViec")
company_names = overview_companies["Company Name"].unique()
selected_companies = st.sidebar.multiselect(
    "Chọn công ty để phân tích/so sánh", company_names, default=[company_names[0]]
)
tab = st.sidebar.radio(
    "Chọn chế độ phân tích",
    ("Tổng quan 1 công ty", "So sánh nhiều công ty", "Phân cụm đánh giá", "Dashboard Radar"),
)

# --- CHỨC NĂNG: PHÂN TÍCH CẢM XÚC 1 CÔNG TY ---
if tab == "Tổng quan 1 công ty" and selected_companies:
    selected_company = selected_companies[0]
    st.header(f"Phân tích cảm xúc review cho {selected_company}")
    company_reviews = reviews[reviews["Company Name"] == selected_company]
    if len(company_reviews) == 0 or company_reviews['What I liked'].notna().sum() == 0:
        st.warning(f"Không có review nào cho công ty {selected_company} hoặc tất cả review đều trống.")
    else:
        filtered_reviews = company_reviews[company_reviews['What I liked'].notna()].copy()
        filtered_reviews['Sentiment'] = sentiment_model.predict(filtered_reviews['What I liked'])
        company_reviews = company_reviews.copy()
        company_reviews.loc[company_reviews['What I liked'].notna(), 'Sentiment'] = filtered_reviews['Sentiment'].values

        st.subheader("Tổng quan cảm xúc")
        sentiment_counts = company_reviews['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        # Wordcloud cho từng nhóm cảm xúc
        for sentiment in ['positive', 'negative', 'neutral']:
            text = " ".join(company_reviews[company_reviews['Sentiment'] == sentiment]['What I liked'].fillna(''))
            if text.strip():
                wc = WordCloud(width=800, height=400, background_color="white").generate(text)
                st.subheader(f"Wordcloud cho review {sentiment}")
                st.image(wc.to_array())

        st.subheader("Các review và cảm xúc")
        st.dataframe(company_reviews[['Title', 'What I liked', 'Sentiment']])

# --- CHỨC NĂNG: SO SÁNH NHIỀU CÔNG TY ---
elif tab == "So sánh nhiều công ty" and selected_companies:
    st.header("So sánh cảm xúc các công ty đã chọn")
    compare_df = reviews[reviews["Company Name"].isin(selected_companies)].copy()
    if compare_df.empty or compare_df['What I liked'].notna().sum() == 0:
        st.warning("Không có đủ dữ liệu review để so sánh.")
    else:
        filtered_reviews = compare_df[compare_df['What I liked'].notna()].copy()
        filtered_reviews['Sentiment'] = sentiment_model.predict(filtered_reviews['What I liked'])
        compare_df.loc[compare_df['What I liked'].notna(), 'Sentiment'] = filtered_reviews['Sentiment'].values

        st.subheader("Bảng so sánh cảm xúc")
        sentiment_counts = (
            compare_df.groupby(['Company Name', 'Sentiment']).size().unstack(fill_value=0)
        )
        st.bar_chart(sentiment_counts)

        st.subheader("Biểu đồ phân bổ cảm xúc theo công ty")
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.subheader("Các review nổi bật từng công ty")
        st.dataframe(compare_df[['Company Name', 'Title', 'What I liked', 'Sentiment']])

# --- CHỨC NĂNG: PHÂN CỤM ĐÁNH GIÁ ---
elif tab == "Phân cụm đánh giá" and selected_companies:
    st.header("Phân cụm đánh giá các công ty đã chọn")
    cluster_labels = {}
    for company in selected_companies:
        idx = overview_companies[overview_companies['Company Name'] == company].index[0]
        label = cluster_model.labels_[idx]
        cluster_labels[company] = label

    cluster_df = pd.DataFrame({
        "Company Name": list(cluster_labels.keys()),
        "Cluster": list(cluster_labels.values()),
        "Top keywords": [cluster_keywords.get(l, []) for l in cluster_labels.values()],
        "Improvement": [cluster_improvement.get(l, "Đang cập nhật...") for l in cluster_labels.values()]
    })

    st.dataframe(cluster_df)

    st.subheader("Tỷ lệ các cụm trên toàn bộ dữ liệu")
    cluster_counts = pd.Series(cluster_model.labels_).value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)

# --- CHỨC NĂNG: DASHBOARD RADAR SO SÁNH ĐA CHIỀU ---
elif tab == "Dashboard Radar" and selected_companies:
    st.header("Dashboard Radar - So sánh đa chiều giữa các công ty")
    metrics = [
        'Salary & benefits',
        'Training & learning',
        'Management cares about me',
        'Culture & fun',
        'Office & workspace'
    ]
    fig = go.Figure()
    for company in selected_companies:
        comp_data = overview_reviews[overview_reviews['Company Name'] == company]
        if not comp_data.empty:
            values = [comp_data[m].values[0] if m in comp_data else 0 for m in metrics]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=company
            ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True
    )
    st.plotly_chart(fig)

    # Bảng tổng hợp chi tiết
    st.subheader("Bảng điểm từng tiêu chí")
    radar_table = overview_reviews[overview_reviews['Company Name'].isin(selected_companies)][['Company Name'] + metrics]
    st.dataframe(radar_table)

# --- Nếu không chọn công ty nào ---
else:
    st.info("Vui lòng chọn ít nhất một công ty để xem kết quả.")

# --- (Tùy chọn) Footer ---
st.markdown(
    "<hr/><center><small>Made by Data Science & ML | ITViec Project 2025</small></center>",
    unsafe_allow_html=True
)
