"""Visual theme shared by NeuroLab pages."""

import streamlit as st


def apply_theme() -> None:
    """Apply NeuroLab's navy, teal, sky blue, beige, and white visual system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --background-color:#F5EFEB; --secondary-background-color:#FFFFFF; --primary-color:#2F4156; --text-color:#2F4156; --bg:#F5EFEB; --surface:#FFFFFF; --surface-raised:#FFFFFF; --surface-soft:#F5EFEB; --line:#C8D9E6; --muted:#567C8D; --text:#2F4156; --plum:#2F4156; --plum-light:#567C8D; --plum-soft:#C8D9E6; --plum-pale:#C8D9E6; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { background:var(--background-color) !important; color:var(--text-color) !important; }
        [data-testid="stHeader"] { background:transparent; color:var(--text); }
        .block-container { max-width:1440px; padding:2rem 2rem 2.5rem; }
        h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--text) !important; letter-spacing:-.035em; }
        p, li, label, [data-testid="stMarkdownContainer"] { color:var(--text); }
        p { color:var(--muted); line-height:1.65; }
        a { color:var(--plum-light) !important; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background:#17263F !important; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div { padding:1.5rem 1rem 1rem; }
        .brand-lockup { display:flex; align-items:center; gap:.78rem; margin:.2rem 0 1.7rem; min-width:0; }
        .brand-mark { flex:0 0 2.8rem; display:grid; place-items:center; width:2.8rem; height:2.8rem; border:1px solid #5B8FF966; border-radius:.9rem; background:linear-gradient(145deg,#182943,#243b5c); box-shadow:0 7px 18px #101a2b55; transition:transform .18s ease,box-shadow .18s ease; }
        .brand-mark:hover { transform:translateY(-1px); box-shadow:0 9px 22px #101a2b77; }
        .brand-mark svg { width:2.1rem; height:2.1rem; overflow:visible; }
        .brand-mark path { fill:none; stroke:#79A7FF; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; opacity:.9; }
        .brand-mark .node { stroke:#DDEBFF; stroke-width:1.5; }
        .brand-mark .node-blue { fill:#5B8FF9; }
        .brand-mark .node-cyan { fill:#4BD4E8; }
        .brand-mark .node-violet { fill:#A78BFA; }
        .brand-copy { min-width:0; }
        .brand-name { color:#FFFFFF; font:700 1.12rem 'Space Grotesk',sans-serif; letter-spacing:-.025em; line-height:1.15; }
        .brand-tagline { margin-top:.3rem; color:#C8D9E6; font-size:.65rem; line-height:1.35; white-space:normal; }
        .sidebar-label { color:#C8D9E6; font-size:.65rem; font-weight:700; letter-spacing:.16em; margin:2.2rem 0 .6rem; }
        [class*="st-key-network_playground_page"] .playground-hero { min-height:245px; margin:.5rem 0 2rem; }
        [class*="st-key-network_playground_page"] .playground-hero { min-height:245px; margin:.5rem 0 1.8rem; }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] { margin:0 0 1rem; border:1px solid #C8D9E6; border-radius:1rem; background:#FFFFFF; box-shadow:0 8px 22px #2F415610; }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] summary p { color:#2F4156 !important; font:700 1rem 'Space Grotesk',sans-serif; }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] summary:hover p { color:#567C8D !important; }
        [class*="st-key-network_playground_page"] [data-testid="stMetric"] { box-shadow:0 4px 14px #2F415610; }
        [class*="st-key-network_playground_page"] [data-testid="stCaptionContainer"] { color:#567C8D; }
        [class*="st-key-play_next_dataset"] button, [class*="st-key-play_next_architecture"] button, [class*="st-key-play_train_button"] button { min-height:3.1rem !important; border:1px solid #567C8D !important; border-radius:.8rem !important; background:linear-gradient(115deg,#2F4156,#3D5D73) !important; color:#FFFFFF !important; font:700 .94rem 'DM Sans',sans-serif !important; letter-spacing:.005em; box-shadow:0 5px 14px #2F415625 !important; transition:transform .18s ease,box-shadow .18s ease,filter .18s ease; }
        [class*="st-key-play_next_dataset"] button *, [class*="st-key-play_next_architecture"] button *, [class*="st-key-play_train_button"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; font-family:'DM Sans',sans-serif !important; font-size:.94rem !important; font-weight:700 !important; opacity:1 !important; }
        [class*="st-key-play_next_dataset"] button:hover, [class*="st-key-play_next_architecture"] button:hover, [class*="st-key-play_train_button"] button:hover { transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 8px 18px #2F415635 !important; }
        [class*="st-key-play_back_architecture"] button, [class*="st-key-play_back_training"] button { min-height:3rem !important; border:1px solid #C8D9E6 !important; border-radius:.8rem !important; background:#FFFFFF !important; color:#2F4156 !important; font:600 .88rem 'DM Sans',sans-serif !important; }
        [class*="st-key-play_back_architecture"] button:hover, [class*="st-key-play_back_training"] button:hover { border-color:#567C8D !important; background:#C8D9E6 !important; }
        [data-testid="stSidebar"] .stButton { margin:.08rem 0; }
        [data-testid="stSidebar"] .stButton > button {
            width:100%; min-height:2.7rem; justify-content:flex-start; text-align:left;
            padding:.58rem .78rem !important; border-radius:.72rem !important;
            font-family:'DM Sans',sans-serif !important; font-size:.92rem !important;
            font-weight:600 !important; letter-spacing:.005em;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
            color:var(--plum-light) !important; background:transparent !important;
            border-color:transparent !important; box-shadow:none !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            color:var(--plum) !important; background:#FFFFFF !important;
            border-color:#FFFFFF !important; transform:translateX(2px);
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            color:#FFFFFF !important; background:var(--plum) !important;
            border-color:var(--plum) !important; box-shadow:0 6px 16px #2F41563b, inset 3px 0 #2F4156 !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] {
            color:#FFFFFF !important;
        }
        .sidebar-footer { position:fixed; bottom:1.4rem; color:#C8D9E6; font-size:.72rem; }
        .status-dot { display:inline-block; width:.48rem; height:.48rem; border-radius:50%; background:#2F4156; margin-right:.45rem; }
        .eyebrow { color:var(--plum-light); font-size:.66rem; font-weight:700; letter-spacing:.18em; margin-bottom:.45rem; }
        .hero { min-height:280px; display:flex; align-items:center; justify-content:space-between; position:relative; overflow:hidden; padding:2.5rem 3rem; border:1px solid #C8D9E6; border-radius:1.25rem; background:linear-gradient(120deg,#FFFFFF 0%,#C8D9E6 100%); box-shadow:0 16px 36px #2F415655; margin:.5rem 0 2.2rem; }
        .hero:after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 83% 48%,#FFFFFF19,transparent 44%); pointer-events:none; }
        .hero-copy { position:relative; z-index:2; max-width:600px; }
        .hero-kicker { color:var(--plum-light); font-size:.68rem; font-weight:700; letter-spacing:.2em; }
        .hero h1 { font-size:clamp(3rem,6vw,4.7rem); line-height:1; margin:.5rem 0 .3rem; }
        .hero-period { color:var(--plum-light); }
        .hero-tagline { color:var(--plum-light); font:600 1.04rem 'Space Grotesk'; margin:.6rem 0; }
        .hero-description { max-width:540px; line-height:1.7; font-size:.94rem; margin:.8rem 0 1.2rem; }
        .hero-chip { display:inline-flex; align-items:center; border:1px solid #FFFFFF; background:#FFFFFFb5; padding:.45rem .75rem; border-radius:2rem; color:var(--plum-light); font-size:.74rem; }
        .hero-chip .status-dot { width:.4rem; height:.4rem; }
        .hero-art { position:absolute; right:5%; top:8%; width:35%; height:84%; opacity:.85; background-image:linear-gradient(28deg,transparent 49.7%,#2F415622 50%,transparent 50.3%),linear-gradient(152deg,transparent 49.7%,#FFFFFF24 50%,transparent 50.3%); }
        .orbit { position:absolute; border:1px solid #2F415625; border-radius:50%; transform:rotate(-25deg); }
        .orbit-one { width:270px;height:150px;top:30px;left:35px; }.orbit-two { width:340px;height:190px;top:10px;left:0; }
        .neuron { position:absolute; width:12px;height:12px;border-radius:50%;background:var(--plum);box-shadow:0 0 0 5px #2F415620;border:2px solid #FFFFFF; }
        .n1{left:3%;top:46%;}.n2{left:25%;top:20%;background:#2F4156;}.n3{left:25%;top:70%;background:#2F4156;}.n4{left:48%;top:10%;}.n5{left:48%;top:43%;background:#FFFFFF;}.n6{left:48%;top:78%;}.n7{left:71%;top:23%;background:#2F4156;}.n8{left:71%;top:66%;background:#2F4156;}.n9{left:94%;top:45%;}.n10{left:24%;top:46%;width:8px;height:8px;}.n11{left:70%;top:46%;width:8px;height:8px;background:#FFFFFF;}
        .section-heading { display:flex; justify-content:space-between; align-items:end; margin:0 0 1rem; }
        .section-heading h2 { font-size:1.5rem; margin:0; }.section-note { color:var(--muted); font-size:.75rem; padding-bottom:.2rem; }
        .feature-card { min-height:178px; box-sizing:border-box; padding:1.2rem 1.25rem; margin-bottom:1rem; border:1px solid var(--line); border-radius:1rem; background:var(--surface); box-shadow:0 8px 20px #2F415630; transition:transform .18s,border-color .18s,box-shadow .18s; }
        .feature-card:hover { transform:translateY(-3px); border-color:#2F4156; box-shadow:0 12px 24px #2F415655; }
        .feature-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:.85rem; }
        .feature-icon { display:grid; place-items:center; width:2.15rem;height:2.15rem;border-radius:.7rem;background:var(--plum-pale);color:var(--plum);font-size:1.05rem; }
        .cyan .feature-icon, .violet .feature-icon { background:var(--plum-pale);color:var(--plum-light); }
        .feature-number { color:#2F4156; font:600 .7rem 'Space Grotesk'; }
        .feature-card h3 { font-size:1rem; margin:0 0 .35rem; letter-spacing:-.02em; }.feature-card p { font-size:.78rem; line-height:1.5; margin:0; min-height:2.2rem; }
        .card-link { margin-top:.65rem; color:var(--plum-light); font-size:.7rem; font-weight:600; }.card-link span { padding-left:.2rem; }
        .info-panel { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.35rem 1.45rem; min-height:164px; box-shadow:0 8px 20px #2F415630; }
        .info-panel h2 { font-size:1.12rem; margin:.2rem 0 .5rem; }.info-panel p { font-size:.8rem; line-height:1.6; margin:0; }
        .progress-track { height:6px; border-radius:5px; background:#FFFFFF; margin-top:1.15rem; overflow:hidden; }.progress-fill { width:24%;height:100%;border-radius:5px;background:var(--plum); }
        .progress-caption { display:flex;justify-content:space-between;color:var(--muted);font-size:.68rem;margin-top:.45rem; }
        .stack-panel { min-height:164px; }.stack-list { display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem; }
        .stack-list span { border:1px solid #FFFFFF;background:#FFFFFF;color:var(--plum-light);padding:.4rem .62rem;border-radius:.55rem;font-size:.72rem;font-weight:600; }
        .page-footer { text-align:center;color:#2F4156;font-size:.67rem;letter-spacing:.08em;margin:2rem 0 .2rem; }.page-footer span { color:var(--plum-light);padding:0 .35rem; }
        .placeholder-card { max-width:680px; margin:2rem auto; text-align:center; padding:3.5rem 2rem; border:1px solid var(--line); border-radius:1.2rem; background:var(--surface); box-shadow:0 14px 32px #2F415655; }
        .placeholder-card h2 { margin:.6rem 0; }.placeholder-card p { margin:0; }.placeholder-icon { display:grid;place-items:center;width:3.2rem;height:3.2rem;margin:auto;border-radius:1rem;background:var(--plum-pale);color:var(--plum);font-size:1.5rem; }
        .stButton > button, [data-testid="stBaseButton-primary"] { border:1px solid var(--plum) !important; border-radius:.7rem !important; background:var(--plum) !important; color:white !important; font-weight:700 !important; transition:all .18s ease; }
        .stButton > button:hover, [data-testid="stBaseButton-primary"]:hover { background:#2F4156 !important; border-color:#2F4156 !important; box-shadow:0 7px 18px #2F415642; }
        [data-testid="stBaseButton-secondary"] { border:1px solid #FFFFFF !important; border-radius:.7rem !important; background:#FFFFFF !important; color:var(--plum) !important; }
        [data-testid="stBaseButton-secondary"]:hover { border-color:var(--plum) !important; background:var(--plum-pale) !important; }
        [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] > div { background:#FFFFFF !important; border-color:#FFFFFF !important; border-radius:.6rem !important; }
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
        [data-testid="stAlert"] { background:#FFFFFF; border-color:#FFFFFF; color:var(--text); }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.7rem; }
        code, [data-testid="stCode"] { background:#FFFFFF !important; color:var(--plum) !important; }
        [data-testid="stSidebar"] .stButton > button {
            background:var(--plum) !important; border-color:var(--plum) !important;
            color:#FFFFFF !important;
        }
        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] .stButton > button p {
            color:#FFFFFF !important; fill:#FFFFFF !important;
            font-family:'DM Sans',sans-serif !important; font-size:.92rem !important;
            font-weight:600 !important; letter-spacing:.005em;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            background:#2F4156 !important; border-color:#2F4156 !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            box-shadow:0 6px 16px #2F415666, inset 3px 0 #2F4156 !important;
        }
        @media (max-width: 800px) { .block-container { padding:1.2rem 1rem 2rem; }.hero { padding:1.7rem; min-height:300px; }.hero-art { opacity:.35; right:0;width:60%; }.section-heading { align-items:start; gap:1rem; flex-direction:column; }.feature-card { min-height:160px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
