"""Visual theme shared by NeuroLab pages."""

import streamlit as st


def apply_theme() -> None:
    """Apply NeuroLab's white and plum visual system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --background-color:#ffffff; --secondary-background-color:#f4f5f8; --primary-color:#35102f; --text-color:#35102f; --bg:#ffffff; --surface:#f4f5f8; --surface-raised:#ffffff; --surface-soft:#ffffff; --line:#c2c8d2; --muted:#716675; --text:#35102f; --plum:#35102f; --plum-light:#65405f; --plum-soft:#8b6c88; --plum-pale:#eee8ef; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { background:var(--background-color) !important; color:var(--text-color) !important; }
        [data-testid="stHeader"] { background:transparent; color:var(--text); }
        .block-container { max-width:1440px; padding:2rem 2rem 2.5rem; }
        h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--text) !important; letter-spacing:-.035em; }
        p, li, label, [data-testid="stMarkdownContainer"] { color:var(--text); }
        p { color:var(--muted); line-height:1.65; }
        a { color:var(--plum-light) !important; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background:#ffffff !important; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div { padding:1.5rem 1rem 1rem; }
        [data-testid="stSidebar"] h3 { margin-top:-3.2rem; margin-left:3.2rem; font-size:1.2rem; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { margin-left:3.2rem; margin-top:-.8rem; font-size:.72rem; color:var(--muted); }
        .brand-mark { display:grid; place-items:center; width:2.4rem; height:2.4rem; border-radius:.8rem; color:white; font:700 1.2rem 'Space Grotesk'; background:var(--plum); box-shadow:0 6px 18px #35102f26; }
        .sidebar-label { color:var(--plum-light); font-size:.65rem; font-weight:700; letter-spacing:.16em; margin:2.2rem 0 .6rem; }
        [data-testid="stSidebar"] .stButton { margin:.08rem 0; }
        [data-testid="stSidebar"] .stButton > button {
            width:100%; min-height:2.7rem; justify-content:flex-start; text-align:left;
            padding:.58rem .78rem !important; border-radius:.72rem !important;
            font-size:.84rem !important; font-weight:600 !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
            color:var(--plum-light) !important; background:transparent !important;
            border-color:transparent !important; box-shadow:none !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            color:var(--plum) !important; background:#eee8ef !important;
            border-color:#e1d8e2 !important; transform:translateX(2px);
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            color:#fff !important; background:var(--plum) !important;
            border-color:var(--plum) !important; box-shadow:0 6px 16px #35102f22, inset 3px 0 #b89bb5 !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] {
            color:#fff !important;
        }
        .sidebar-footer { position:fixed; bottom:1.4rem; color:var(--muted); font-size:.72rem; }
        .status-dot { display:inline-block; width:.48rem; height:.48rem; border-radius:50%; background:#72aa96; margin-right:.45rem; }
        .eyebrow { color:var(--plum-light); font-size:.66rem; font-weight:700; letter-spacing:.18em; margin-bottom:.45rem; }
        .hero { min-height:280px; display:flex; align-items:center; justify-content:space-between; position:relative; overflow:hidden; padding:2.5rem 3rem; border:1px solid #c8c0cc; border-radius:1.25rem; background:linear-gradient(120deg,#f6f6f8 0%,#e9e8ee 100%); box-shadow:0 16px 36px #35102f0b; margin:.5rem 0 2.2rem; }
        .hero:after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 83% 48%,#8b6c8819,transparent 44%); pointer-events:none; }
        .hero-copy { position:relative; z-index:2; max-width:600px; }
        .hero-kicker { color:var(--plum-light); font-size:.68rem; font-weight:700; letter-spacing:.2em; }
        .hero h1 { font-size:clamp(3rem,6vw,4.7rem); line-height:1; margin:.5rem 0 .3rem; }
        .hero-period { color:var(--plum-light); }
        .hero-tagline { color:var(--plum-light); font:600 1.04rem 'Space Grotesk'; margin:.6rem 0; }
        .hero-description { max-width:540px; line-height:1.7; font-size:.94rem; margin:.8rem 0 1.2rem; }
        .hero-chip { display:inline-flex; align-items:center; border:1px solid #d2cbd5; background:#ffffffb5; padding:.45rem .75rem; border-radius:2rem; color:var(--plum-light); font-size:.74rem; }
        .hero-chip .status-dot { width:.4rem; height:.4rem; }
        .hero-art { position:absolute; right:5%; top:8%; width:35%; height:84%; opacity:.85; background-image:linear-gradient(28deg,transparent 49.7%,#65405f22 50%,transparent 50.3%),linear-gradient(152deg,transparent 49.7%,#8b6c8824 50%,transparent 50.3%); }
        .orbit { position:absolute; border:1px solid #65405f25; border-radius:50%; transform:rotate(-25deg); }
        .orbit-one { width:270px;height:150px;top:30px;left:35px; }.orbit-two { width:340px;height:190px;top:10px;left:0; }
        .neuron { position:absolute; width:12px;height:12px;border-radius:50%;background:var(--plum);box-shadow:0 0 0 5px #35102f12;border:2px solid #fff; }
        .n1{left:3%;top:46%;}.n2{left:25%;top:20%;background:#65405f;}.n3{left:25%;top:70%;background:#65405f;}.n4{left:48%;top:10%;}.n5{left:48%;top:43%;background:#8b6c88;}.n6{left:48%;top:78%;}.n7{left:71%;top:23%;background:#65405f;}.n8{left:71%;top:66%;background:#65405f;}.n9{left:94%;top:45%;}.n10{left:24%;top:46%;width:8px;height:8px;}.n11{left:70%;top:46%;width:8px;height:8px;background:#8b6c88;}
        .section-heading { display:flex; justify-content:space-between; align-items:end; margin:0 0 1rem; }
        .section-heading h2 { font-size:1.5rem; margin:0; }.section-note { color:var(--muted); font-size:.75rem; padding-bottom:.2rem; }
        .feature-card { min-height:178px; box-sizing:border-box; padding:1.2rem 1.25rem; margin-bottom:1rem; border:1px solid var(--line); border-radius:1rem; background:var(--surface); box-shadow:0 8px 20px #35102f08; transition:transform .18s,border-color .18s,box-shadow .18s; }
        .feature-card:hover { transform:translateY(-3px); border-color:#a88fa6; box-shadow:0 12px 24px #35102f10; }
        .feature-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:.85rem; }
        .feature-icon { display:grid; place-items:center; width:2.15rem;height:2.15rem;border-radius:.7rem;background:var(--plum-pale);color:var(--plum);font-size:1.05rem; }
        .cyan .feature-icon, .violet .feature-icon { background:var(--plum-pale);color:var(--plum-light); }
        .feature-number { color:#988e9a; font:600 .7rem 'Space Grotesk'; }
        .feature-card h3 { font-size:1rem; margin:0 0 .35rem; letter-spacing:-.02em; }.feature-card p { font-size:.78rem; line-height:1.5; margin:0; min-height:2.2rem; }
        .card-link { margin-top:.65rem; color:var(--plum-light); font-size:.7rem; font-weight:600; }.card-link span { padding-left:.2rem; }
        .info-panel { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.35rem 1.45rem; min-height:164px; box-shadow:0 8px 20px #35102f08; }
        .info-panel h2 { font-size:1.12rem; margin:.2rem 0 .5rem; }.info-panel p { font-size:.8rem; line-height:1.6; margin:0; }
        .progress-track { height:6px; border-radius:5px; background:#d5d2da; margin-top:1.15rem; overflow:hidden; }.progress-fill { width:24%;height:100%;border-radius:5px;background:var(--plum); }
        .progress-caption { display:flex;justify-content:space-between;color:var(--muted);font-size:.68rem;margin-top:.45rem; }
        .stack-panel { min-height:164px; }.stack-list { display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem; }
        .stack-list span { border:1px solid #d3ccd6;background:#eeebf0;color:var(--plum-light);padding:.4rem .62rem;border-radius:.55rem;font-size:.72rem;font-weight:600; }
        .page-footer { text-align:center;color:#857a88;font-size:.67rem;letter-spacing:.08em;margin:2rem 0 .2rem; }.page-footer span { color:var(--plum-light);padding:0 .35rem; }
        .placeholder-card { max-width:680px; margin:2rem auto; text-align:center; padding:3.5rem 2rem; border:1px solid var(--line); border-radius:1.2rem; background:var(--surface); box-shadow:0 14px 32px #35102f0b; }
        .placeholder-card h2 { margin:.6rem 0; }.placeholder-card p { margin:0; }.placeholder-icon { display:grid;place-items:center;width:3.2rem;height:3.2rem;margin:auto;border-radius:1rem;background:var(--plum-pale);color:var(--plum);font-size:1.5rem; }
        .stButton > button, [data-testid="stBaseButton-primary"] { border:1px solid var(--plum) !important; border-radius:.7rem !important; background:var(--plum) !important; color:white !important; font-weight:700 !important; transition:all .18s ease; }
        .stButton > button:hover, [data-testid="stBaseButton-primary"]:hover { background:#4a2143 !important; border-color:#4a2143 !important; box-shadow:0 7px 18px #35102f25; }
        [data-testid="stBaseButton-secondary"] { border:1px solid #b9afbd !important; border-radius:.7rem !important; background:#f4f5f8 !important; color:var(--plum) !important; }
        [data-testid="stBaseButton-secondary"]:hover { border-color:var(--plum) !important; background:var(--plum-pale) !important; }
        [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] > div { background:#f8f8fa !important; border-color:#c8c2cc !important; border-radius:.6rem !important; }
        [data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="select"] input { color:var(--text) !important; }
        [data-baseweb="select"] svg { fill:var(--plum-light) !important; }
        [data-testid="stSlider"] [role="slider"] { background:var(--plum) !important; border-color:var(--plum) !important; }
        [data-testid="stProgressBar"] > div > div { background:var(--plum) !important; }
        [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:.8rem; padding:.85rem 1rem; }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:var(--text) !important; }
        [data-testid="stExpander"] { background:var(--surface); border:1px solid var(--line); border-radius:.8rem; }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:var(--text) !important; }
        [data-testid="stTabs"] [role="tab"] { color:var(--muted); }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--plum) !important; }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--plum) !important; }
        [data-testid="stAlert"] { background:#f4f1f5; border-color:#c9bdce; color:var(--text); }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.7rem; }
        code, [data-testid="stCode"] { background:#eeebf0 !important; color:var(--plum) !important; }
        [data-testid="stSidebar"] .stButton > button {
            background:var(--plum) !important; border-color:var(--plum) !important;
            color:#fff !important;
        }
        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] .stButton > button p {
            color:#fff !important; fill:#fff !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            background:#4a2143 !important; border-color:#4a2143 !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            box-shadow:0 6px 16px #35102f30, inset 3px 0 #d4b8d1 !important;
        }
        @media (max-width: 800px) { .block-container { padding:1.2rem 1rem 2rem; }.hero { padding:1.7rem; min-height:300px; }.hero-art { opacity:.35; right:0;width:60%; }.section-heading { align-items:start; gap:1rem; flex-direction:column; }.feature-card { min-height:160px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
