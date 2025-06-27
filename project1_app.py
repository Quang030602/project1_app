import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
# pip install streamlit-aggrid
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
# pip install -r requirements.txt

# ---------- CSS CHO TOÀN BỘ APP: TÔNG XANH DƯƠNG - TRẮNG + GRADIENT + CARD + BUTTON ---------- #
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);
    color: #1e3a8a;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(115deg, #1e40af 20%, #3b82f6 100%);
    color: #fff;
}
.stSidebar .sidebar-content { color: #fff; }
h1, h2, h3, .stApp header { color: #1e40af !important; }
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #1e40af, #3b82f6, #dbeafe);
    margin: 20px 0;
}
.stButton>button {
    color: #1e40af;
    background: #ffffff;
    border-radius: 8px;
    border: 2px solid #3b82f6;
    transition: all 0.3s ease;
    font-weight: 600;
}
.stButton>button:hover {
    background: #3b82f6;
    color: #fff;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.4);
}
.stDataFrame, .stTable, .stAgGrid {
    background: #ffffff !important;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    transition: all 0.3s ease;
}
.stDataFrame:hover, .stTable:hover, .stAgGrid:hover {
    box-shadow: 0 6px 20px rgba(59,130,246,0.2);
    transform: translateY(-2px);
    border-color: #3b82f6;
}
/* Nút bấm gradient */
.blue-btn {
    display: inline-block;
    padding: 12px 28px;
    font-size: 16px;
    font-weight: 600;
    color: #fff !important;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    border: none;
    border-radius: 25px;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3);
    cursor: pointer;
    margin: 12px 10px 12px 0;
    transition: all 0.3s ease;
}
.blue-btn:hover {
    box-shadow: 0 8px 25px rgba(59,130,246,0.4);
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
    transform: translateY(-2px) scale(1.02);
}
/* Card tổng quan, cluster, section-box */
.summary-card, .section-box, .cluster-card {
    border-radius: 15px;
    padding: 20px 28px;
    margin: 16px 0 20px 0;
    box-shadow: 0 4px 20px rgba(59,130,246,0.15);
    transition: all 0.3s ease;
    border: 1px solid rgba(59,130,246,0.1);
}
.summary-card:hover, .section-box:hover, .cluster-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.25);
}
.summary-card {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: #fff;
    border-left: 6px solid #1d4ed8;
    font-size: 18px;
    font-weight: 500;
}
.summary-card-title {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #ffffff;
    margin-bottom: 8px;
}
.section-box {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    color: #1e3a8a;
    border-left: 4px solid #3b82f6;
}
.cluster-card {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    color: #1e40af;
    font-size: 16px;
    font-weight: 500;
    border-left: 4px solid #60a5fa;
}
.cluster-card:hover {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
}
/* Hover cho keyword tags */
.keyword-tag {
    background: #dbeafe;
    color: #1e40af;
    border-radius: 6px;
    padding: 4px 10px;
    margin: 2px 4px 2px 0;
    display: inline-block;
    transition: all 0.2s ease;
    font-weight: 500;
    font-size: 14px;
}
.keyword-tag:hover {
    background: #3b82f6;
    color: #fff;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ---------- #
@st.cache_data
def load_data():
    reviews = pd.read_excel("Du lieu cung cap/Reviews.xlsx")
    overview_companies = pd.read_excel("Du lieu cung cap/Overview_Companies.xlsx")
    overview_reviews = pd.read_excel("Du lieu cung cap/Overview_Reviews.xlsx")
    overview_reviews = overview_reviews.rename(columns={"id": "company_id"})
    overview_companies = overview_companies.rename(columns={"id": "company_id"})
    data = reviews.merge(overview_reviews[['company_id', 'Overall rating']], left_on='id', right_on='company_id', how='left') \
                  .merge(overview_companies[['company_id', 'Company Name', 'Company Type', 'Company size']], on='company_id', how='left')
    data = data.rename(columns={"Company Name_y": "Company Name"})
    data = data.dropna(subset=['What I liked'])
    return data, overview_companies, overview_reviews

data, overview_companies, overview_reviews = load_data()

sentiment_model = joblib.load("sentiment_model.pkl")
cluster_model = joblib.load("cluster_model.pkl")
cluster_keywords = joblib.load("cluster_keywords.pkl")

cluster_improvement = {
    0: "Tăng cường training, lộ trình phát triển cá nhân rõ ràng.",
    1: "Cải thiện phúc lợi, xem xét tăng lương và chế độ đãi ngộ.",
    2: "Xây dựng văn hóa doanh nghiệp gắn kết, đa dạng hoạt động team-building.",
    3: "Tăng cường truyền thông nội bộ, chú trọng feedback nhân viên.",
}

cluster_names = {
    0: "Văn phòng đẹp & Môi trường tốt",
    1: "Phúc lợi tốt",
    2: "Cần cải thiện đào tạo",
    3: "Truyền thông nội bộ tốt"
}

st.sidebar.title("Phân tích dữ liệu ITViec")
company_names = data["Company Name"].dropna().unique()
selected_companies = st.sidebar.multiselect(
    "Chọn công ty để phân tích/so sánh", company_names, default=[company_names[0]]
)

# Bộ lọc nâng cao
unique_years = data['Year'].dropna().unique() if 'Year' in data.columns else []
unique_positions = data['Position'].dropna().unique() if 'Position' in data.columns else []
unique_sentiments = ['positive (tích cực)', 'neutral (trung tính)', 'negative (tiêu cực)']

with st.sidebar.expander("🎯 Bộ lọc nâng cao"):
    selected_year = st.selectbox("Năm", ["Tất cả"] + sorted(map(str, unique_years))) if len(unique_years) > 1 else "Tất cả"
    selected_position = st.selectbox("Chức vụ", ["Tất cả"] + list(unique_positions)) if len(unique_positions) > 1 else "Tất cả"
    selected_sentiment = st.selectbox("Cảm xúc", ["Tất cả"] + unique_sentiments)

tab = st.sidebar.radio(
    "Chọn chế độ phân tích",
    ("Tổng quan 1 công ty", "So sánh nhiều công ty", "Phân cụm đánh giá", "Dashboard Radar"),
)

# Apply filter function
def apply_filters(data_to_filter):
    filtered_data = data_to_filter.copy()
    if selected_year != "Tất cả" and 'Year' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Year'].astype(str) == selected_year]
    if selected_position != "Tất cả" and 'Position' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['Position'] == selected_position]
    return filtered_data

# ---------- TAB: TỔNG QUAN 1 CÔNG TY ---------- #
if tab == "Tổng quan 1 công ty" and selected_companies:
    selected_company = selected_companies[0]
    st.header(f"Phân tích cảm xúc review cho {selected_company}")
    company_reviews = data[data["Company Name"] == selected_company]
    
    # Apply filters
    company_reviews = apply_filters(company_reviews)
    
    if company_reviews.empty:
        st.warning(f"Không có review nào cho công ty {selected_company} với bộ lọc đã chọn.")
    else:
        filtered_reviews = company_reviews[company_reviews['What I liked'].notna()].copy()
        filtered_reviews['Sentiment'] = sentiment_model.predict(filtered_reviews['What I liked'])
        company_reviews = company_reviews.copy()
        company_reviews.loc[company_reviews['What I liked'].notna(), 'Sentiment'] = filtered_reviews['Sentiment'].values

        sentiment_map = {
            2: "positive (tích cực)",
            1: "neutral (trung tính)",
            0: "negative (tiêu cực)"
        }
        df_show = company_reviews[['What I liked', 'Sentiment']].copy()
        df_show['Cảm xúc đánh giá'] = df_show['Sentiment'].map(sentiment_map)
        df_show = df_show[['What I liked', 'Cảm xúc đánh giá']]

        sentiment_counts = df_show['Cảm xúc đánh giá'].value_counts()
        # Card tổng quan
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-card-title">Tổng quan cảm xúc {selected_company}</div>
            <ul>
                <li>Review tích cực: <b>{sentiment_counts.get('positive (tích cực)',0)}</b></li>
                <li>Review trung tính: <b>{sentiment_counts.get('neutral (trung tính)',0)}</b></li>
                <li>Review tiêu cực: <b>{sentiment_counts.get('negative (tiêu cực)',0)}</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Nút bấm với chức năng thực tế
        if st.button("🔍 Xem đề xuất cải thiện", key="improvement_btn"):
            # Tính tỷ lệ cảm xúc
            total_reviews = len(df_show)
            positive_ratio = sentiment_counts.get('positive (tích cực)', 0) / total_reviews
            negative_ratio = sentiment_counts.get('negative (tiêu cực)', 0) / total_reviews
            neutral_ratio = sentiment_counts.get('neutral (trung tính)', 0) / total_reviews
            
            # Đề xuất dựa trên phân tích cảm xúc
            improvements = []
            
            if negative_ratio > 0.3:  # Nếu > 30% review tiêu cực
                improvements.append("🔴 **Ưu tiên cao**: Cần cải thiện ngay các vấn đề gây bức xúc cho nhân viên")
                improvements.append("📞 Tổ chức các buổi listening session để lắng nghe phản hồi trực tiếp")
                
            if positive_ratio < 0.4:  # Nếu < 40% review tích cực
                improvements.append("⚠️ **Cần chú ý**: Tăng cường các yếu tố tạo sự hài lòng cho nhân viên")
                improvements.append("🎯 Xây dựng chương trình recognition & reward rõ ràng")
                
            if neutral_ratio > 0.4:  # Nếu > 40% review trung tính
                improvements.append("📈 **Cơ hội phát triển**: Nhiều nhân viên đang ở trạng thái trung lập")
                improvements.append("💡 Tạo thêm các hoạt động engagement để nâng cao trải nghiệm")
                
            # Đề xuất chung
            improvements.extend([
                "🏢 **Môi trường làm việc**: Đầu tư cải thiện không gian làm việc và tiện ích",
                "📚 **Đào tạo & phát triển**: Xây dựng lộ trình career path rõ ràng",
                "💰 **Phúc lợi**: Review và cải thiện gói lương thưởng, benefits",
                "🤝 **Quản lý**: Tăng cường training cho management team về leadership skills",
                "📱 **Truyền thông nội bộ**: Cải thiện kênh thông tin và feedback hai chiều"
            ])
            
            # Hiển thị đề xuất
            st.markdown("""
            <div class="section-box">
                <h3>🎯 Đề xuất cải thiện cho """ + selected_company + """</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for improvement in improvements:
                st.markdown(f"• {improvement}")
                
            # Thêm phân tích từ keywords nếu có
            if selected_company in [overview_companies.iloc[i]['Company Name'] for i in range(len(overview_companies))]:
                company_idx = overview_companies[overview_companies['Company Name'] == selected_company].index[0]
                if company_idx < len(cluster_model.labels_):
                    cluster_id = cluster_model.labels_[company_idx]
                    if cluster_id in cluster_keywords:
                        st.markdown(f"""
                        <div class="cluster-card">
                            <b>🔍 Phân tích dựa trên clustering:</b><br>
                            <b>Nhóm:</b> {cluster_names.get(cluster_id, f'Cluster {cluster_id}')}<br>
                            <b>Đặc điểm chính:</b> {', '.join(cluster_keywords[cluster_id])}<br>
                            <b>Hướng cải thiện:</b> {cluster_improvement.get(cluster_id, 'Đang phân tích...')}
                        </div>
                        """, unsafe_allow_html=True)

        # Section review + AgGrid
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Các review và cảm xúc")
        
        # AgGrid configuration
        gb = GridOptionsBuilder.from_dataframe(df_show)
        gb.configure_default_column(editable=False, filter=True, sortable=True, resizable=True)
        gb.configure_pagination(paginationAutoPageSize=True)
        
        # Tooltip cho từng cell
        cell_tooltip = JsCode("""
        function(params) {
            if(params.colDef.field === 'Cảm xúc đánh giá') {
                return {'value': 'Phân loại: ' + params.value};
            }
            return {'value': params.value};
        }
        """)
        
        gb.configure_column('What I liked', tooltipField='What I liked', cellRenderer=cell_tooltip, width=400)
        gb.configure_column('Cảm xúc đánh giá', tooltipField='Cảm xúc đánh giá', cellRenderer=cell_tooltip, width=200)
        gridOptions = gb.build()

        AgGrid(df_show, gridOptions=gridOptions, enable_enterprise_modules=False, 
               height=350, fit_columns_on_grid_load=True, theme='streamlit')
        st.markdown('</div>', unsafe_allow_html=True)

        # Bar chart section
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Biểu đồ tổng quan cảm xúc")
        st.bar_chart(sentiment_counts)
        st.markdown('</div>', unsafe_allow_html=True)

        # Wordcloud từng nhóm cảm xúc, colormap xanh dương
        for sentiment_text, sentiment_label in [
            ('positive', "Tích cực"),
            ('negative', "Tiêu cực"),
            ('neutral', "Trung tính")
        ]:
            text = " ".join(df_show[df_show['Cảm xúc đánh giá'].str.contains(sentiment_text, case=False)]['What I liked'].fillna(''))
            if text.strip():
                wc = WordCloud(width=800, height=400, background_color="white", colormap="Blues").generate(text)
                st.markdown('<div class="section-box">', unsafe_allow_html=True)
                st.subheader(f"Wordcloud cho review {sentiment_label}")
                st.image(wc.to_array())
                st.markdown('</div>', unsafe_allow_html=True)

# ---------- TAB: SO SÁNH NHIỀU CÔNG TY ---------- #
elif tab == "So sánh nhiều công ty" and selected_companies:
    st.header("So sánh cảm xúc các công ty đã chọn")
    compare_df = data[data["Company Name"].isin(selected_companies)].copy()
    
    # Apply filters
    compare_df = apply_filters(compare_df)
    
    if compare_df.empty:
        st.warning("Không có đủ dữ liệu review để so sánh với bộ lọc đã chọn.")
    else:
        filtered_reviews = compare_df[compare_df['What I liked'].notna()].copy()
        filtered_reviews['Sentiment'] = sentiment_model.predict(filtered_reviews['What I liked'])
        compare_df.loc[compare_df['What I liked'].notna(), 'Sentiment'] = filtered_reviews['Sentiment'].values

        sentiment_map = {
            2: "positive (tích cực)",
            1: "neutral (trung tính)",
            0: "negative (tiêu cực)"
        }
        compare_df['Cảm xúc đánh giá'] = compare_df['Sentiment'].map(sentiment_map)

        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Bảng so sánh cảm xúc")
        sentiment_counts = (
            compare_df.groupby(['Company Name', 'Cảm xúc đánh giá']).size().unstack(fill_value=0)
        )
        st.bar_chart(sentiment_counts)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Biểu đồ phân bổ cảm xúc theo công ty")
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Các review nổi bật từng công ty (AgGrid)")
        
        # AgGrid for comparison
        compare_display = compare_df[['Company Name', 'What I liked', 'Cảm xúc đánh giá']].copy()
        gb_compare = GridOptionsBuilder.from_dataframe(compare_display)
        gb_compare.configure_default_column(editable=False, filter=True, sortable=True, resizable=True)
        gb_compare.configure_pagination(paginationAutoPageSize=True)
        gb_compare.configure_column('Company Name', pinned='left', width=150)
        gb_compare.configure_column('What I liked', width=400)
        gb_compare.configure_column('Cảm xúc đánh giá', width=180)
        gridOptions_compare = gb_compare.build()

        AgGrid(compare_display, gridOptions=gridOptions_compare, enable_enterprise_modules=False, 
               height=400, fit_columns_on_grid_load=True, theme='streamlit')
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- TAB: PHÂN CỤM ĐÁNH GIÁ ---------- #
elif tab == "Phân cụm đánh giá" and selected_companies:
    st.header("Phân cụm đánh giá các công ty đã chọn")
    cluster_labels = {}
    for company in selected_companies:
        idx = overview_companies[overview_companies['Company Name'] == company].index[0]
        label = cluster_model.labels_[idx]
        cluster_labels[company] = label

    cluster_df = pd.DataFrame({
        "Company Name": list(cluster_labels.keys()),
        "Cụm ý nghĩa": [cluster_names.get(l, f"Cluster {l}") for l in cluster_labels.values()],
        "Top keywords": [cluster_keywords.get(l, []) for l in cluster_labels.values()],
        "Improvement": [cluster_improvement.get(l, "Đang cập nhật...") for l in cluster_labels.values()]
    })

    for i, row in cluster_df.iterrows():
        st.markdown(f"""
        <div class="cluster-card">
            <b>{row['Company Name']}</b><br>
            <b>Nhóm:</b> {row['Cụm ý nghĩa']}<br>
            <b>Top keywords:</b> {" ".join(f"<span class='keyword-tag'>{k}</span>" for k in row['Top keywords'])}
            <br>
            <b>Đề xuất cải thiện:</b> {row['Improvement']}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Tỷ lệ các cụm trên toàn bộ dữ liệu")
    cluster_counts = pd.Series(cluster_model.labels_).value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    colors = ['#2193b0', '#6dd5ed', '#76a9e7', '#b3e1ff']
    ax2.pie(cluster_counts, labels=[cluster_names.get(i, f"Cluster {i}") for i in cluster_counts.index], 
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.axis("equal")
    st.pyplot(fig2)

# ---------- TAB: DASHBOARD RADAR ---------- #
elif tab == "Dashboard Radar" and selected_companies:
    st.header("Dashboard Radar - So sánh đa chiều giữa các công ty")
    metrics = [
        'Salary & benefits',
        'Training & learning',
        'Management cares about me',
        'Culture & fun',
        'Office & workspace'
    ]
    
    # Responsive container với font adaptive
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        for company in selected_companies:
            comp_data = overview_reviews[overview_reviews['Company Name'] == company]
            if not comp_data.empty:
                values = [comp_data[m].values[0] if m in comp_data else 0 for m in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=company,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 '%{theta}: %{r}/5<br>' +
                                 '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="rgba(32,103,178,0.8)",
                        font=dict(color="rgba(23,64,109,1)", size=14)
                    )
                ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 5],
                    gridcolor="rgba(32,103,178,0.3)",
                    linecolor="rgba(32,103,178,0.5)",
                    tickfont=dict(color="black", size=12)
                ),
                angularaxis=dict(
                    gridcolor="rgba(32,103,178,0.3)",
                    linecolor="rgba(32,103,178,0.5)",
                    tickfont=dict(color="black", size=14)
                )
            ),
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="black", size=12),
            autosize=True,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Chú thích tiêu chí")
        st.markdown("""
        **📊 Thang điểm**: 0-5  
        **💰 Salary & benefits**: Lương, thưởng, phúc lợi  
        **📚 Training & learning**: Đào tạo, phát triển  
        **🤝 Management**: Quan tâm của quản lý  
        **🎉 Culture & fun**: Văn hóa, hoạt động  
        **🏢 Office & workspace**: Không gian làm việc
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Bảng điểm từng tiêu chí (AgGrid)")
    radar_table = overview_reviews[overview_reviews['Company Name'].isin(selected_companies)][['Company Name'] + metrics]
    radar_table = radar_table.reset_index(drop=True)
    
    # AgGrid cho radar table
    gb_radar = GridOptionsBuilder.from_dataframe(radar_table)
    gb_radar.configure_default_column(editable=False, filter=True, sortable=True, resizable=True)
    
    # Tooltip cho từng cell
    cell_tooltip_radar = JsCode("""
    function(params) {
        var tooltips = {
            'Salary & benefits': 'Lương, thưởng, phúc lợi',
            'Training & learning': 'Cơ hội đào tạo, phát triển kỹ năng',
            'Management cares about me': 'Quản lý quan tâm tới nhân viên',
            'Culture & fun': 'Văn hóa, hoạt động tập thể',
            'Office & workspace': 'Không gian làm việc, cơ sở vật chất'
        };
        var tooltip = tooltips[params.colDef.field] || '';
        return {'value': params.value + ' điểm - ' + tooltip};
    }
    """)
    
    gb_radar.configure_column('Company Name', pinned='left', width=180)
    for col in metrics:
        gb_radar.configure_column(col, tooltipField=col, cellRenderer=cell_tooltip_radar, width=120)
    
    gridOptions_radar = gb_radar.build()
    
    AgGrid(radar_table, gridOptions=gridOptions_radar, enable_enterprise_modules=False, 
           height=300, fit_columns_on_grid_load=True, theme='streamlit')
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Vui lòng chọn ít nhất một công ty để xem kết quả.")

st.markdown(
    "<hr/><center><small>Made by Data Science & ML | ITViec Project 2025</small></center>",
    unsafe_allow_html=True
)
