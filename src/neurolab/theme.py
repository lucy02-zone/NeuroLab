"""Visual theme shared by NeuroLab pages."""

import streamlit as st


def apply_theme() -> None:
    """Apply the dark navy visual system and reusable component styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --bg:#080d19; --panel:#101827; --panel2:#0d1523; --line:#1d2a3d; --muted:#8b9bb2; --text:#eef4ff; --blue:#5687ff; --cyan:#39d5e7; --violet:#a17bff; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        .stApp { background:radial-gradient(ellipse at 78% 0%,#14203b 0%,var(--bg) 42%); color:var(--text); }
        .block-container { max-width:1440px; padding-top:2rem; padding-bottom:2rem; }
        h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--text) !important; letter-spacing:-.035em; }
        p { color:#9eacc0; }
        [data-testid="stSidebar"] { background:#0a111e; border-right:1px solid #1b2739; }
        [data-testid="stSidebar"] > div { padding-top:1.5rem; }
        [data-testid="stSidebar"] h3 { margin-top:-3.2rem; margin-left:3.2rem; font-size:1.2rem; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { margin-left:3.2rem; margin-top:-.8rem; font-size:.72rem; }
        .brand-mark { display:grid; place-items:center; width:2.4rem; height:2.4rem; border-radius:.8rem; color:white; font:700 1.2rem 'Space Grotesk'; background:linear-gradient(135deg,#427fff,#36d3e1); box-shadow:0 5px 22px #3978ff44; }
        .sidebar-label { color:#65768f; font-size:.65rem; font-weight:700; letter-spacing:.16em; margin:2.2rem 0 .6rem; }
        [data-testid="stRadio"] > label { display:none; }
        [data-testid="stRadio"] [role="radiogroup"] { gap:.22rem; }
        [data-testid="stRadio"] [role="radiogroup"] label { color:#9eacc0; border-radius:.65rem; padding:.62rem .7rem; transition:all .18s; }
        [data-testid="stRadio"] [role="radiogroup"] label:hover { background:#131f32; color:#fff; }
        [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { background:linear-gradient(90deg,#1b2d4b,#17243a); color:#dce9ff; }
        [data-testid="stRadio"] [role="radiogroup"] label p { font-size:.83rem; }
        .sidebar-footer { position:fixed; bottom:1.4rem; color:#718198; font-size:.72rem; }
        .status-dot { display:inline-block; width:.48rem; height:.48rem; border-radius:50%; background:#43d9b1; margin-right:.45rem; box-shadow:0 0 12px #43d9b188; }
        .eyebrow { color:#6f8db8; font-size:.66rem; font-weight:700; letter-spacing:.18em; margin-bottom:.45rem; }
        .hero { min-height:280px; display:flex; align-items:center; justify-content:space-between; position:relative; overflow:hidden; padding:2.4rem 3rem; border:1px solid #223149; border-radius:1.25rem; background:linear-gradient(112deg,#111c2d 0%,#111c30 55%,#171d3b 100%); box-shadow:0 22px 70px #02061166; margin:.5rem 0 2.4rem; }
        .hero:after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 78% 52%,#546cff18,transparent 42%); pointer-events:none; }
        .hero-copy { position:relative; z-index:2; max-width:590px; }
        .hero-kicker { color:#83a8ff; font-size:.68rem; font-weight:700; letter-spacing:.2em; }
        .hero h1 { font-size:clamp(3rem,6vw,4.7rem); line-height:1; margin:.5rem 0 .3rem; }
        .hero-period { color:var(--cyan); }
        .hero-tagline { color:#dae6ff; font:500 1.04rem 'Space Grotesk'; margin:.6rem 0; }
        .hero-description { max-width:520px; line-height:1.65; font-size:.91rem; margin:.8rem 0 1.2rem; }
        .hero-chip { display:inline-flex; align-items:center; border:1px solid #2b3b53; background:#0b1422a8; padding:.45rem .7rem; border-radius:2rem; color:#b9c7dc; font-size:.72rem; }
        .hero-chip .status-dot { width:.4rem; height:.4rem; }
        .hero-art { position:absolute; right:6%; top:8%; width:34%; height:84%; opacity:.9; background-image:linear-gradient(28deg,transparent 49.7%,#528bff33 50%,transparent 50.3%),linear-gradient(152deg,transparent 49.7%,#46dce133 50%,transparent 50.3%); }
        .orbit { position:absolute; border:1px solid #698cff22; border-radius:50%; transform:rotate(-25deg); }
        .orbit-one { width:270px;height:150px;top:30px;left:35px; }.orbit-two { width:340px;height:190px;top:10px;left:0; }
        .neuron { position:absolute; width:12px;height:12px;border-radius:50%;background:#7b9cff;box-shadow:0 0 18px #6b8effaa;border:2px solid #c6d4ff; }
        .n1{left:3%;top:46%;}.n2{left:25%;top:20%;background:#42dce8;}.n3{left:25%;top:70%;background:#42dce8;}.n4{left:48%;top:10%;}.n5{left:48%;top:43%;background:#ae88ff;}.n6{left:48%;top:78%;}.n7{left:71%;top:23%;background:#42dce8;}.n8{left:71%;top:66%;background:#42dce8;}.n9{left:94%;top:45%;}.n10{left:24%;top:46%;width:8px;height:8px;}.n11{left:70%;top:46%;width:8px;height:8px;background:#ae88ff;}
        .section-heading { display:flex; justify-content:space-between; align-items:end; margin:0 0 1rem; }
        .section-heading h2 { font-size:1.5rem; margin:0; }.section-note { color:#718198; font-size:.75rem; padding-bottom:.2rem; }
        .feature-card { height:178px; box-sizing:border-box; padding:1.15rem 1.25rem; margin-bottom:1rem; border:1px solid var(--line); border-radius:1rem; background:linear-gradient(145deg,#111b2a,#0e1725); transition:transform .18s,border-color .18s; }
        .feature-card:hover { transform:translateY(-3px); border-color:#344866; }
        .feature-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:.85rem; }
        .feature-icon { display:grid; place-items:center; width:2rem;height:2rem;border-radius:.65rem;background:#1b3150;color:#79a5ff;font-size:1.05rem; }
        .cyan .feature-icon { background:#12323a;color:#5be1e8; }.violet .feature-icon { background:#29213f;color:#b59aff; }
        .feature-number { color:#485970; font:600 .7rem 'Space Grotesk'; }
        .feature-card h3 { font-size:1rem; margin:0 0 .35rem; letter-spacing:-.02em; }.feature-card p { font-size:.76rem; line-height:1.48; margin:0; min-height:2.2rem; }
        .card-link { margin-top:.65rem; color:#83a7ee; font-size:.69rem; }.card-link span { padding-left:.2rem; }
        .info-panel { background:linear-gradient(145deg,#111b2a,#0d1624); border:1px solid var(--line); border-radius:1rem; padding:1.3rem 1.45rem; min-height:164px; }
        .info-panel h2 { font-size:1.12rem; margin:.2rem 0 .5rem; }.info-panel p { font-size:.78rem; line-height:1.55; margin:0; }
        .progress-track { height:5px; border-radius:5px; background:#1c2b3e; margin-top:1.15rem; overflow:hidden; }.progress-fill { width:24%;height:100%;border-radius:5px;background:linear-gradient(90deg,#5488ff,#45d9df); }
        .progress-caption { display:flex;justify-content:space-between;color:#73849c;font-size:.68rem;margin-top:.4rem; }
        .stack-panel { min-height:164px; }.stack-list { display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem; }
        .stack-list span { border:1px solid #26364b;background:#131f30;color:#afc0d7;padding:.38rem .58rem;border-radius:.5rem;font-size:.7rem; }
        .page-footer { text-align:center;color:#56677f;font-size:.65rem;letter-spacing:.08em;margin:2rem 0 .2rem; }.page-footer span { color:#4ecddd;padding:0 .35rem; }
        .placeholder-card { max-width:680px; margin:2rem auto; text-align:center; padding:3.5rem 2rem; border:1px solid #223149; border-radius:1.2rem; background:linear-gradient(145deg,#111b2a,#0d1523); }
        .placeholder-card h2 { margin:.6rem 0; }.placeholder-card p { margin:0; }.placeholder-icon { display:grid;place-items:center;width:3rem;height:3rem;margin:auto;border-radius:1rem;background:#1a2d4a;color:#74a1ff;font-size:1.5rem; }
        @media (max-width: 800px) { .hero { padding:1.7rem; min-height:300px; }.hero-art { opacity:.23; right:0;width:60%; }.section-heading { align-items:start; gap:1rem; flex-direction:column; }.feature-card { height:auto; min-height:170px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
