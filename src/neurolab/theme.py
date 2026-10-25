"""Shared visual system for NeuroLab's Streamlit pages."""

import streamlit as st


def apply_theme() -> None:
    """Apply NeuroLab's navy, blue, and teal interface theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root {
            --background-color:#F3F6FA;
            --secondary-background-color:#FFFFFF;
            --primary-color:#2F5F83;
            --text-color:#17263F;
            --page:#F3F6FA;
            --surface:#FFFFFF;
            --surface-alt:#EAF1F7;
            --surface-raised:#FFFFFF;
            --line:#D8E2EC;
            --ink:#17263F;
            --muted:#61758B;
            --navy:#17263F;
            --blue:#4E73D8;
            --teal:#397F8B;
            --cyan:#49B7C4;
            --violet:#8B79CA;
            --plum:#2F5F83;
            --plum-light:#397F8B;
            --plum-soft:#DDEAF3;
            --plum-pale:#EAF1F7;
        }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] {
            background:var(--page) !important; color:var(--ink) !important;
        }
        [data-testid="stHeader"] { background:transparent; }
        .block-container { max-width:1440px; padding:2rem 2rem 2.75rem; }
        h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em; }
        h1 { font-weight:700 !important; }
        p, li, label { color:var(--ink); }
        p { color:var(--muted); line-height:1.65; }
        a { color:var(--teal) !important; }

        /* Sidebar brand and navigation */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
            background:var(--navy) !important; border-right:1px solid #263B56;
        }
        [data-testid="stSidebar"] > div { padding:1.35rem .9rem 1rem; }
        .brand-lockup { display:flex; align-items:center; gap:.78rem; margin:.2rem 0 1.55rem; min-width:0; }
        .brand-mark { flex:0 0 2.8rem; display:grid; place-items:center; width:2.8rem; height:2.8rem; border:1px solid #7098E866; border-radius:.9rem; background:linear-gradient(145deg,#233C5E,#315982); box-shadow:0 7px 18px #08152555; transition:transform .18s ease,box-shadow .18s ease; }
        .brand-mark:hover { transform:translateY(-1px); box-shadow:0 9px 22px #08152577; }
        .brand-mark svg { width:2.1rem; height:2.1rem; overflow:visible; }
        .brand-mark path { fill:none; stroke:#8DB8FF; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; opacity:.95; }
        .brand-mark .node { stroke:#E6F0FF; stroke-width:1.5; }
        .brand-mark .node-blue { fill:var(--blue); }
        .brand-mark .node-cyan { fill:var(--cyan); }
        .brand-mark .node-violet { fill:var(--violet); }
        .brand-copy { min-width:0; }
        .brand-name { color:#FFFFFF; font:700 1.12rem 'Space Grotesk',sans-serif; letter-spacing:-.025em; line-height:1.15; }
        .brand-tagline { margin-top:.3rem; color:#C2D0E0; font-size:.65rem; line-height:1.35; white-space:normal; }
        .sidebar-label { color:#92A9C2; font-size:.64rem; font-weight:700; letter-spacing:.16em; margin:1.7rem 0 .65rem; }
        [data-testid="stSidebar"] .stButton { margin:.16rem 0; }
        [data-testid="stSidebar"] .stButton > button {
            width:100%; min-height:2.55rem; justify-content:flex-start; text-align:left;
            padding:.55rem .72rem !important; border:1px solid transparent !important; border-radius:.72rem !important;
            background:transparent !important; color:#D5E0EC !important; box-shadow:none !important;
            font:600 .86rem 'DM Sans',sans-serif !important; transition:background .16s ease,border-color .16s ease,color .16s ease;
        }
        [data-testid="stSidebar"] .stButton > button * { color:inherit !important; fill:currentColor !important; font:inherit !important; }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            background:#213650 !important; border-color:#314B6B !important; color:#FFFFFF !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            background:#2A4566 !important; border-color:#426B91 !important; color:#FFFFFF !important;
            box-shadow:inset 3px 0 var(--cyan),0 5px 14px #08152533 !important;
        }
        .sidebar-footer { position:fixed; bottom:1.15rem; color:#91A5BC; font-size:.7rem; }

        /* Shared page headings, hero, cards, and status elements */
        .eyebrow { color:var(--teal); font-size:.65rem; font-weight:700; letter-spacing:.18em; margin-bottom:.45rem; }
        .hero { min-height:270px; display:flex; align-items:center; justify-content:space-between; position:relative; overflow:hidden; padding:2.35rem 2.7rem; border:1px solid #D9E4EF; border-radius:1.25rem; background:linear-gradient(120deg,#FFFFFF 0%,#E5EEF7 100%); box-shadow:0 12px 30px #1B35500D; margin:.5rem 0 2rem; }
        .hero:after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at 84% 48%,#8B79CA12,transparent 45%); pointer-events:none; }
        .hero-copy { position:relative; z-index:2; max-width:620px; }
        .hero-kicker { color:var(--teal); font-size:.66rem; font-weight:700; letter-spacing:.2em; }
        .hero h1 { font-size:clamp(2.7rem,5.5vw,4.5rem); line-height:1.02; margin:.55rem 0 .3rem; }
        .hero-period { color:var(--cyan); }
        .hero-tagline { color:var(--navy); font:600 1rem 'Space Grotesk',sans-serif; margin:.65rem 0; }
        .hero-description { max-width:560px; line-height:1.7; font-size:.91rem; margin:.75rem 0 1rem; }
        .hero-chip { display:inline-flex; align-items:center; border:1px solid #D8E2EC; background:#FFFFFFD9; padding:.43rem .72rem; border-radius:2rem; color:var(--muted); font-size:.72rem; }
        .status-dot { display:inline-block; width:.46rem; height:.46rem; border-radius:50%; background:var(--teal); margin-right:.45rem; }
        .hero-chip .status-dot { width:.4rem; height:.4rem; }
        .hero-art { position:absolute; right:5%; top:8%; width:35%; height:84%; opacity:.78; background-image:linear-gradient(28deg,transparent 49.7%,#49B7C41C 50%,transparent 50.3%),linear-gradient(152deg,transparent 49.7%,#8B79CA18 50%,transparent 50.3%); }
        .orbit { position:absolute; border:1px solid #397F8B25; border-radius:50%; transform:rotate(-25deg); }
        .orbit-one { width:270px;height:150px;top:30px;left:35px; }.orbit-two { width:340px;height:190px;top:10px;left:0; }
        .neuron { position:absolute; width:12px;height:12px;border-radius:50%;background:var(--blue);box-shadow:0 0 0 5px #4E73D81A;border:2px solid #FFFFFF; }
        .n1{left:3%;top:46%;}.n2{left:25%;top:20%;background:var(--cyan);}.n3{left:25%;top:70%;background:var(--cyan);}.n4{left:48%;top:10%;}.n5{left:48%;top:43%;background:var(--violet);}.n6{left:48%;top:78%;}.n7{left:71%;top:23%;background:var(--cyan);}.n8{left:71%;top:66%;background:var(--cyan);}.n9{left:94%;top:45%;}.n10{left:24%;top:46%;width:8px;height:8px;}.n11{left:70%;top:46%;width:8px;height:8px;background:var(--violet);}
        .section-heading { display:flex; justify-content:space-between; align-items:end; margin:0 0 1rem; }
        .section-heading h2 { font-size:1.45rem; margin:0; }.section-note { color:var(--muted); font-size:.74rem; padding-bottom:.2rem; }
        .feature-card { min-height:174px; box-sizing:border-box; padding:1.15rem 1.2rem; margin-bottom:1rem; border:1px solid var(--line); border-radius:1rem; background:var(--surface); box-shadow:0 5px 16px #1B355008; transition:transform .18s,border-color .18s,box-shadow .18s; }
        .feature-card:hover { transform:translateY(-2px); border-color:#AFC6D9; box-shadow:0 10px 24px #1B355010; }
        .feature-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:.8rem; }
        .feature-icon { display:grid; place-items:center; width:2.1rem;height:2.1rem;border-radius:.7rem;background:var(--surface-alt);color:var(--blue);font-size:1.02rem; }
        .cyan .feature-icon { background:#E3F3F4;color:var(--teal); }.violet .feature-icon { background:#F0EDFA;color:var(--violet); }
        .feature-number { color:#8294A9; font:600 .68rem 'Space Grotesk',sans-serif; }
        .feature-card h3 { font-size:.98rem; margin:0 0 .35rem; letter-spacing:-.02em; }.feature-card p { font-size:.77rem; line-height:1.5; margin:0; min-height:2.2rem; }
        .card-link { margin-top:.65rem; color:var(--teal); font-size:.7rem; font-weight:700; }.card-link span { padding-left:.2rem; }
        .info-panel { background:var(--surface); border:1px solid var(--line); border-radius:1rem; padding:1.3rem 1.4rem; min-height:158px; box-shadow:0 5px 16px #1B355008; }
        .info-panel h2 { font-size:1.08rem; margin:.2rem 0 .5rem; }.info-panel p { font-size:.79rem; line-height:1.6; margin:0; }
        .progress-track { height:6px; border-radius:5px; background:#E5ECF3; margin-top:1.1rem; overflow:hidden; }.progress-fill { width:24%;height:100%;border-radius:5px;background:linear-gradient(90deg,var(--blue),var(--cyan)); }
        .progress-caption { display:flex;justify-content:space-between;color:var(--muted);font-size:.68rem;margin-top:.45rem; }
        .stack-panel { min-height:158px; }.stack-list { display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem; }
        .stack-list span { border:1px solid var(--line);background:#F7FAFC;color:var(--teal);padding:.4rem .62rem;border-radius:.55rem;font-size:.72rem;font-weight:600; }
        .page-footer { text-align:center;color:var(--muted);font-size:.67rem;letter-spacing:.08em;margin:2rem 0 .2rem; }.page-footer span { color:var(--teal);padding:0 .35rem; }
        .placeholder-card { max-width:680px; margin:2rem auto; text-align:center; padding:3rem 2rem; border:1px solid var(--line); border-radius:1.2rem; background:var(--surface); box-shadow:0 10px 26px #1B35500A; }
        .placeholder-card h2 { margin:.6rem 0; }.placeholder-card p { margin:0; }.placeholder-icon { display:grid;place-items:center;width:3.1rem;height:3.1rem;margin:auto;border-radius:1rem;background:var(--surface-alt);color:var(--blue);font-size:1.4rem; }

        /* Inputs and action buttons */
        .stButton > button, [data-testid="stBaseButton-primary"] { min-height:2.65rem; border:1px solid var(--navy) !important; border-radius:.72rem !important; background:var(--navy) !important; color:#FFFFFF !important; font:700 .88rem 'DM Sans',sans-serif !important; transition:transform .16s ease,background .16s ease,box-shadow .16s ease; }
        .stButton > button *, [data-testid="stBaseButton-primary"] * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; font:inherit !important; }
        .stButton > button:hover, [data-testid="stBaseButton-primary"]:hover { transform:translateY(-1px); background:#294866 !important; border-color:#294866 !important; box-shadow:0 6px 14px #17263F20 !important; }
        [data-testid="stBaseButton-secondary"] { min-height:2.55rem; border:1px solid var(--line) !important; border-radius:.72rem !important; background:#FFFFFF !important; color:var(--navy) !important; font:600 .86rem 'DM Sans',sans-serif !important; }
        [data-testid="stBaseButton-secondary"] * { color:var(--navy) !important; -webkit-text-fill-color:var(--navy) !important; font:inherit !important; }
        [data-testid="stBaseButton-secondary"]:hover { border-color:#AFC6D9 !important; background:#F5F9FC !important; }
        [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] > div { background:#FFFFFF !important; border:1px solid var(--line) !important; border-radius:.65rem !important; }
        [data-baseweb="input"] > div:focus-within, [data-baseweb="select"] > div:focus-within, [data-baseweb="textarea"] > div:focus-within { border-color:var(--blue) !important; box-shadow:0 0 0 3px #4E73D81A !important; }
        [data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="select"] input { color:var(--ink) !important; }
        [data-baseweb="select"] svg { fill:var(--muted) !important; }
        [data-testid="stSlider"] [role="slider"] { background:var(--blue) !important; border-color:#FFFFFF !important; box-shadow:0 0 0 1px var(--blue); }
        [data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,var(--blue),var(--cyan)) !important; }
        [data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--line); border-radius:.85rem; padding:.85rem 1rem; box-shadow:0 4px 14px #1B355008; }
        [data-testid="stMetricLabel"] { color:var(--muted) !important; }
        [data-testid="stMetricValue"] { color:var(--navy) !important; font-family:'Space Grotesk',sans-serif; }
        [data-testid="stExpander"] { background:#FFFFFF; border:1px solid var(--line); border-radius:.85rem; box-shadow:0 4px 14px #1B355006; }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:var(--ink) !important; }
        [data-testid="stTabs"] [role="tab"] { color:var(--muted); font-weight:600; }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--navy) !important; }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--teal) !important; }
        [data-testid="stAlert"] { background:#FFFFFF; border:1px solid var(--line); border-radius:.8rem; color:var(--ink); }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.75rem; overflow:hidden; }
        code, [data-testid="stCode"] { background:#EAF1F7 !important; color:var(--navy) !important; border-radius:.3rem; }

        /* Step-by-step Network Playground */
        [class*="st-key-network_playground_page"] .playground-hero { min-height:245px; margin:.5rem 0 1.6rem; }
        [class*="st-key-network_playground_page"] [data-testid="stProgressBar"] { margin:.3rem 0 1.15rem; }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] { margin:0 0 .85rem; border-color:var(--line); border-radius:1rem; background:#FFFFFF; box-shadow:0 5px 17px #1B355008; }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] summary p { color:var(--navy) !important; font:700 .98rem 'Space Grotesk',sans-serif; }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] summary:hover p { color:var(--teal) !important; }
        [class*="st-key-network_playground_page"] [data-testid="stMetric"] { box-shadow:0 4px 14px #1B355008; }
        [class*="st-key-network_playground_page"] [data-testid="stCaptionContainer"] { color:var(--muted); }
        [class*="st-key-play_next_dataset"] button, [class*="st-key-play_next_architecture"] button, [class*="st-key-play_train_button"] button { min-height:3rem !important; border:1px solid #2F5F83 !important; border-radius:.78rem !important; background:linear-gradient(115deg,#17263F,#2F5F83) !important; color:#FFFFFF !important; font:700 .92rem 'DM Sans',sans-serif !important; box-shadow:0 6px 15px #17263F20 !important; transition:transform .16s ease,box-shadow .16s ease,filter .16s ease; }
        [class*="st-key-play_next_dataset"] button *, [class*="st-key-play_next_architecture"] button *, [class*="st-key-play_train_button"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; font:inherit !important; opacity:1 !important; }
        [class*="st-key-play_next_dataset"] button:hover, [class*="st-key-play_next_architecture"] button:hover, [class*="st-key-play_train_button"] button:hover { transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 9px 20px #17263F28 !important; }
        [class*="st-key-play_back_architecture"] button, [class*="st-key-play_back_training"] button { min-height:3rem !important; border:1px solid var(--line) !important; border-radius:.78rem !important; background:#FFFFFF !important; color:var(--navy) !important; }
        [class*="st-key-play_back_architecture"] button *, [class*="st-key-play_back_training"] button * { color:var(--navy) !important; -webkit-text-fill-color:var(--navy) !important; }
        [class*="st-key-play_back_architecture"] button:hover, [class*="st-key-play_back_training"] button:hover { border-color:#AFC6D9 !important; background:#F5F9FC !important; }

        /* Dataset Explorer follows the same guided layout and visual rhythm. */
        [class*="st-key-dataset_explorer_page"] .playground-hero { min-height:245px; margin:.5rem 0 1.6rem; }
        [class*="st-key-dataset_explorer_page"] [data-testid="stProgressBar"] { margin:.3rem 0 1.15rem; }
        [class*="st-key-dataset_explorer_page"] [data-testid="stExpander"] { margin:0 0 .85rem; border-color:var(--line); border-radius:1rem; background:#FFFFFF; box-shadow:0 5px 17px #1B355008; }
        [class*="st-key-dataset_explorer_page"] [data-testid="stExpander"] summary p { color:var(--navy) !important; font:700 .98rem 'Space Grotesk',sans-serif; }
        [class*="st-key-dataset_explorer_page"] [data-testid="stExpander"] summary:hover p { color:var(--teal) !important; }
        [class*="st-key-dataset_explorer_page"] [data-testid="stMetric"] { box-shadow:0 4px 14px #1B355008; }
        [class*="st-key-dataset_explorer_page"] [data-testid="stCaptionContainer"] { color:var(--muted); }
        [class*="st-key-explorer_next_setup"] button, [class*="st-key-explorer_next_analysis"] button, [class*="st-key-explorer_transfer"] button { min-height:3rem !important; border:1px solid #2F5F83 !important; border-radius:.78rem !important; background:linear-gradient(115deg,#17263F,#2F5F83) !important; color:#FFFFFF !important; font:700 .92rem 'DM Sans',sans-serif !important; box-shadow:0 6px 15px #17263F20 !important; transition:transform .16s ease,box-shadow .16s ease,filter .16s ease; }
        [class*="st-key-explorer_next_setup"] button *, [class*="st-key-explorer_next_analysis"] button *, [class*="st-key-explorer_transfer"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; font:inherit !important; opacity:1 !important; }
        [class*="st-key-explorer_next_setup"] button:hover, [class*="st-key-explorer_next_analysis"] button:hover, [class*="st-key-explorer_transfer"] button:hover { transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 9px 20px #17263F28 !important; }
        [class*="st-key-explorer_back_analysis"] button, [class*="st-key-explorer_back_split"] button { min-height:3rem !important; border:1px solid var(--line) !important; border-radius:.78rem !important; background:#FFFFFF !important; color:var(--navy) !important; }
        [class*="st-key-explorer_back_analysis"] button *, [class*="st-key-explorer_back_split"] button * { color:var(--navy) !important; -webkit-text-fill-color:var(--navy) !important; }
        [class*="st-key-explorer_back_analysis"] button:hover, [class*="st-key-explorer_back_split"] button:hover { border-color:#AFC6D9 !important; background:#F5F9FC !important; }

        /* Model Manager shares the Network Playground's guided page treatment. */
        [class*="st-key-model_manager_page"] .playground-hero { min-height:245px; margin:.5rem 0 1.6rem; }
        [class*="st-key-model_manager_page"] [data-testid="stProgressBar"] { margin:.3rem 0 1.15rem; }
        [class*="st-key-model_manager_page"] [data-testid="stExpander"] { margin:0 0 .85rem; border-color:var(--line); border-radius:1rem; background:#FFFFFF; box-shadow:0 5px 17px #1B355008; }
        [class*="st-key-model_manager_page"] [data-testid="stExpander"] summary p { color:var(--navy) !important; font:700 .98rem 'Space Grotesk',sans-serif; }
        [class*="st-key-model_manager_page"] [data-testid="stExpander"] summary:hover p { color:var(--teal) !important; }
        [class*="st-key-model_manager_page"] [data-testid="stMetric"] { box-shadow:0 4px 14px #1B355008; }
        [class*="st-key-model_manager_page"] [data-testid="stCaptionContainer"] { color:var(--muted); }
        [class*="st-key-model_next_export"] button, [class*="st-key-model_next_import"] button, [class*="st-key-model_import_confirm"] button, [class*="st-key-model_prediction_run"] button, [class*="st-key-model_export_download"] button { min-height:3rem !important; border:1px solid #2F5F83 !important; border-radius:.78rem !important; background:linear-gradient(115deg,#17263F,#2F5F83) !important; color:#FFFFFF !important; font:700 .92rem 'DM Sans',sans-serif !important; box-shadow:0 6px 15px #17263F20 !important; transition:transform .16s ease,box-shadow .16s ease,filter .16s ease; }
        [class*="st-key-model_next_export"] button *, [class*="st-key-model_next_import"] button *, [class*="st-key-model_import_confirm"] button *, [class*="st-key-model_prediction_run"] button *, [class*="st-key-model_export_download"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; font:inherit !important; opacity:1 !important; }
        [class*="st-key-model_next_export"] button:hover, [class*="st-key-model_next_import"] button:hover, [class*="st-key-model_import_confirm"] button:hover, [class*="st-key-model_prediction_run"] button:hover, [class*="st-key-model_export_download"] button:hover { transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 9px 20px #17263F28 !important; }
        [class*="st-key-model_back_import"] button, [class*="st-key-model_back_review"] button { min-height:3rem !important; border:1px solid var(--line) !important; border-radius:.78rem !important; background:#FFFFFF !important; color:var(--navy) !important; }
        [class*="st-key-model_back_import"] button *, [class*="st-key-model_back_review"] button * { color:var(--navy) !important; -webkit-text-fill-color:var(--navy) !important; }
        [class*="st-key-model_back_import"] button:hover, [class*="st-key-model_back_review"] button:hover { border-color:#AFC6D9 !important; background:#F5F9FC !important; }

        /* Experiment Wizard: the same hero, progress, and polished action system. */
        [class*="st-key-experiment_wizard_page"] .playground-hero { min-height:245px; margin:.5rem 0 1.6rem; }
        [class*="st-key-experiment_wizard_page"] [data-testid="stProgressBar"] { margin:.3rem 0 1.15rem; }
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] { margin:.5rem 0 1.1rem; padding:1.25rem 1.35rem; border-color:var(--line); border-radius:1rem; background:#FFFFFF; box-shadow:0 5px 17px #1B355008; }
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] [data-baseweb="input"] > div,
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] [data-baseweb="textarea"] > div,
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] [data-baseweb="select"] > div { background:#F7FAFD !important; border:1.5px solid #AFC3D6 !important; border-radius:.68rem !important; box-shadow:0 1px 2px #17263F0A; }
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] [data-baseweb="input"] > div:focus-within,
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] [data-baseweb="textarea"] > div:focus-within,
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] [data-baseweb="select"] > div:focus-within { background:#FFFFFF !important; border-color:var(--blue) !important; box-shadow:0 0 0 3px #4E73D826 !important; }
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] input,
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] textarea { color:var(--navy) !important; }
        /* Keep the three experiment-detail fields visibly outlined at rest. */
        [class*="st-key-wizard_name"] [data-baseweb="input"] > div,
        [class*="st-key-wizard_seed"] [data-baseweb="input"] > div,
        [class*="st-key-wizard_description"] [data-baseweb="textarea"] > div,
        [class*="st-key-wizard_notes"] [data-baseweb="textarea"] > div { background:#F7FAFD !important; border:2px solid #9CB4CB !important; border-radius:.68rem !important; box-shadow:0 2px 6px #17263F0D !important; }
        [class*="st-key-wizard_name"] input,
        [class*="st-key-wizard_seed"] input,
        [class*="st-key-wizard_description"] textarea,
        [class*="st-key-wizard_notes"] textarea { background:transparent !important; border:0 !important; border-radius:inherit !important; color:#17263F !important; box-shadow:none !important; }
        [class*="st-key-wizard_name"]:focus-within [data-baseweb="input"] > div,
        [class*="st-key-wizard_seed"]:focus-within [data-baseweb="input"] > div,
        [class*="st-key-wizard_description"]:focus-within [data-baseweb="textarea"] > div,
        [class*="st-key-wizard_notes"]:focus-within [data-baseweb="textarea"] > div { background:#FFFFFF !important; border-color:#4E73D8 !important; box-shadow:0 0 0 3px #4E73D826 !important; }
        [class*="st-key-experiment_wizard_page"] [class*="st-key-wizard_step_panel"] label { color:#40566F !important; font-weight:600 !important; }
        .wizard-step-heading { display:flex; align-items:center; gap:.85rem; margin:.2rem 0 .8rem; }
        .wizard-step-heading > span { padding:.38rem .62rem; border-radius:2rem; background:#EAF1F7; color:var(--teal); font:700 .65rem 'DM Sans',sans-serif; letter-spacing:.12em; }
        .wizard-step-heading h2 { margin:0; color:var(--navy) !important; font:700 1.2rem 'Space Grotesk',sans-serif; }
        .wizard-step-count { margin-left:auto; color:var(--muted); font:600 .76rem 'Space Grotesk',sans-serif; }
        [class*="st-key-experiment_wizard_page"] [data-testid="stCaptionContainer"] { color:var(--muted); }
        [class*="st-key-wizard_next_"] button, [class*="st-key-wizard_confirm"] button, [class*="st-key-wizard_train"] button, [class*="st-key-wizard_to_evaluate"] button, [class*="st-key-wizard_eval_bottom"] button, [class*="st-key-wizard_save_bottom"] button, [class*="st-key-wizard_save"] button, [class*="st-key-wizard_finish"] button { min-height:3rem !important; border:1px solid #2F5F83 !important; border-radius:.78rem !important; background:linear-gradient(115deg,#17263F,#2F5F83) !important; color:#FFFFFF !important; font:700 .92rem 'DM Sans',sans-serif !important; box-shadow:0 6px 15px #17263F20 !important; transition:transform .16s ease,box-shadow .16s ease,filter .16s ease; }
        [class*="st-key-wizard_next_"] button *, [class*="st-key-wizard_confirm"] button *, [class*="st-key-wizard_train"] button *, [class*="st-key-wizard_to_evaluate"] button *, [class*="st-key-wizard_eval_bottom"] button *, [class*="st-key-wizard_save_bottom"] button *, [class*="st-key-wizard_save"] button *, [class*="st-key-wizard_finish"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; font:inherit !important; opacity:1 !important; }
        [class*="st-key-wizard_next_"] button:hover, [class*="st-key-wizard_confirm"] button:hover, [class*="st-key-wizard_train"] button:hover, [class*="st-key-wizard_to_evaluate"] button:hover, [class*="st-key-wizard_eval_bottom"] button:hover, [class*="st-key-wizard_save_bottom"] button:hover, [class*="st-key-wizard_save"] button:hover, [class*="st-key-wizard_finish"] button:hover { transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 9px 20px #17263F28 !important; }
        [class*="st-key-wizard_back_"] button, [class*="st-key-wizard_reset"] button, [class*="st-key-wizard_open_reports"] button { min-height:2.8rem !important; border:1px solid var(--line) !important; border-radius:.78rem !important; background:#FFFFFF !important; color:var(--navy) !important; }
        [class*="st-key-wizard_back_"] button *, [class*="st-key-wizard_reset"] button *, [class*="st-key-wizard_open_reports"] button * { color:var(--navy) !important; -webkit-text-fill-color:var(--navy) !important; }
        [class*="st-key-wizard_back_"] button:hover, [class*="st-key-wizard_reset"] button:hover, [class*="st-key-wizard_open_reports"] button:hover { border-color:#AFC6D9 !important; background:#F5F9FC !important; }

        /* Neural Math Calculator: clear guide steps, tabs, and editable fields. */
        [class*="st-key-math_calculator_page"] .playground-hero { min-height:245px; margin:.5rem 0 1.6rem; }
        [class*="st-key-math_calculator_page"] [data-testid="stProgressBar"] { margin:.3rem 0 1rem; }
        .math-guide-note { display:flex; align-items:center; gap:.75rem; margin:.2rem 0 .65rem; padding:.85rem 1rem; border:1px solid #C9D9E8; border-left:4px solid var(--teal); border-radius:.78rem; background:#EAF1F7; color:var(--navy); font-size:.86rem; line-height:1.5; }
        .math-guide-note span { flex:0 0 auto; color:var(--teal); font:700 .65rem 'DM Sans',sans-serif; letter-spacing:.12em; }
        [class*="st-key-math_calculator_page"] [data-testid="stTabs"] [data-baseweb="tab-list"] { display:flex; gap:.3rem; padding:.35rem; border:1px solid var(--line); border-radius:.85rem; background:#EAF1F7; }
        [class*="st-key-math_calculator_page"] [data-testid="stTabs"] [role="tab"] { justify-content:center; min-height:2.7rem; padding:.55rem .7rem; border:1px solid transparent; border-radius:.62rem; color:#536A82 !important; font-size:.79rem; transition:background .15s,border-color .15s,color .15s,box-shadow .15s; }
        [class*="st-key-math_calculator_page"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] { border-color:#D3DFEA; background:#FFFFFF; color:var(--navy) !important; box-shadow:0 2px 7px #17263F12; }
        [class*="st-key-math_calculator_page"] [data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none; }
        [class*="st-key-math_calculator_page"] [data-testid="stTabs"] [role="tabpanel"] { padding:1rem .15rem .25rem; }
        [class*="st-key-math_calculator_page"] [data-baseweb="input"] > div,
        [class*="st-key-math_calculator_page"] [data-baseweb="textarea"] > div,
        [class*="st-key-math_calculator_page"] [data-baseweb="select"] > div { background:#F7FAFD !important; border:1.5px solid #AFC3D6 !important; border-radius:.68rem !important; box-shadow:0 1px 2px #17263F0A; }
        [class*="st-key-math_calculator_page"] [data-baseweb="input"] > div:focus-within,
        [class*="st-key-math_calculator_page"] [data-baseweb="textarea"] > div:focus-within,
        [class*="st-key-math_calculator_page"] [data-baseweb="select"] > div:focus-within { background:#FFFFFF !important; border-color:var(--blue) !important; box-shadow:0 0 0 3px #4E73D826 !important; }
        [class*="st-key-math_calculator_page"] [data-baseweb="input"] input,
        [class*="st-key-math_calculator_page"] [data-baseweb="textarea"] textarea { color:var(--navy) !important; }
        [class*="st-key-math_calculator_page"] [data-testid="stCaptionContainer"] { color:var(--muted); }
        [class*="st-key-mathcalc_guide_next"] button { min-height:2.8rem !important; border:1px solid #2F5F83 !important; border-radius:.75rem !important; background:linear-gradient(115deg,#17263F,#2F5F83) !important; color:#FFFFFF !important; font-weight:700 !important; box-shadow:0 5px 13px #17263F20 !important; }
        [class*="st-key-mathcalc_guide_next"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }
        [class*="st-key-mathcalc_guide_back"] button { min-height:2.8rem !important; border:1px solid var(--line) !important; border-radius:.75rem !important; background:#FFFFFF !important; color:var(--navy) !important; }
        [class*="st-key-mathcalc_guide_back"] button * { color:var(--navy) !important; -webkit-text-fill-color:var(--navy) !important; }

        /* Quiz Center: visible quiz identity, question cards, and selectable answers. */
        [class*="st-key-quiz_center_page"] .quiz-hero { min-height:245px; margin:.5rem 0 1.6rem; }
        .quiz-section-heading { margin:1.35rem 0 .8rem; }
        .quiz-section-heading span { color:var(--teal); font:700 .63rem 'DM Sans',sans-serif; letter-spacing:.17em; }
        .quiz-section-heading h2 { margin:.22rem 0 0; color:var(--navy) !important; font:700 1.3rem 'Space Grotesk',sans-serif; }
        [class*="st-key-quiz_center_page"] [data-testid="stMetric"] { min-height:5.4rem; }
        [class*="st-key-quiz_center_page"] [data-baseweb="select"] > div { background:#F7FAFD !important; border:1.5px solid #AFC3D6 !important; border-radius:.68rem !important; }
        [class*="st-key-quiz_center_page"] [data-baseweb="select"] > div:focus-within { background:#FFFFFF !important; border-color:var(--blue) !important; box-shadow:0 0 0 3px #4E73D826 !important; }
        [class*="st-key-quiz_question_card"] { margin:1rem 0 1.25rem; padding:1.4rem 1.55rem; border-color:#C9D9E8; border-radius:1.05rem; background:linear-gradient(145deg,#FFFFFF 0%,#F8FBFE 100%); box-shadow:0 9px 24px #17263F0C; }
        .quiz-question-meta { display:flex; align-items:center; gap:.55rem; flex-wrap:wrap; margin:0 0 .7rem; }
        .quiz-question-number { padding:.38rem .58rem; border-radius:2rem; background:#E8F0F8; color:#2F5F83; font:700 .63rem 'DM Sans',sans-serif; letter-spacing:.1em; }
        .quiz-question-category { color:var(--muted); font-size:.74rem; font-weight:600; }
        .quiz-difficulty { margin-left:auto; padding:.34rem .58rem; border:1px solid #D8E2EC; border-radius:2rem; background:#FFFFFF; color:#536A82; font-size:.68rem; font-weight:700; }
        .quiz-difficulty.difficulty-beginner { border-color:#B8DCCD; background:#EDF7F2; color:#287253; }
        .quiz-difficulty.difficulty-intermediate { border-color:#D8D0A9; background:#FAF7E9; color:#79651D; }
        .quiz-difficulty.difficulty-advanced { border-color:#DCC9D2; background:#F8EFF3; color:#80506A; }
        .quiz-question-prompt { margin:.8rem 0 1rem; color:var(--navy); font:700 clamp(1.15rem,2vw,1.55rem)/1.45 'Space Grotesk',sans-serif; letter-spacing:-.02em; }
        [class*="st-key-quiz_question_card"] [data-testid="stProgressBar"] { margin:.1rem 0 1rem; }
        [class*="st-key-quiz_question_card"] [data-testid="stRadio"] div[role="radiogroup"] { gap:.45rem; }
        [class*="st-key-quiz_question_card"] [data-testid="stRadio"] div[role="radiogroup"] > label { min-height:2.8rem; padding:.65rem .85rem; border:1px solid #D4E0EB; border-radius:.72rem; background:#FFFFFF; color:var(--navy) !important; transition:border-color .15s,background .15s,box-shadow .15s; }
        [class*="st-key-quiz_question_card"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover { border-color:#91ACC6; background:#F3F8FC; }
        [class*="st-key-quiz_question_card"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) { border-color:var(--blue); background:#EDF3FC; box-shadow:0 0 0 2px #4E73D81A; }
        [class*="st-key-quiz_question_card"] [data-baseweb="input"] > div { background:#F7FAFD !important; border:1.5px solid #AFC3D6 !important; border-radius:.68rem !important; }
        [class*="st-key-quiz_question_card"] [data-baseweb="input"] > div:focus-within { border-color:var(--blue) !important; box-shadow:0 0 0 3px #4E73D826 !important; }
        [class*="st-key-quiz_submit_"] button, [class*="st-key-quiz_next_"] button { min-height:2.9rem !important; padding:.55rem 1.15rem !important; border:1px solid #2F5F83 !important; border-radius:.75rem !important; background:linear-gradient(115deg,#17263F,#2F5F83) !important; color:#FFFFFF !important; font-weight:700 !important; box-shadow:0 5px 13px #17263F20 !important; }
        [class*="st-key-quiz_submit_"] button *, [class*="st-key-quiz_next_"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; }
        [class*="st-key-quiz_question_card"] [data-testid="stAlert"] { margin:.8rem 0; }
        @media (max-width:800px) { [class*="st-key-quiz_question_card"] { padding:1rem; } .quiz-difficulty { margin-left:0; } }

        /* Optimizer Lab uses the same structured experiment-page design. */
        [class*="st-key-optimizer_lab_page"] .playground-hero { min-height:245px; margin:.5rem 0 1.6rem; }
        .optimizer-section-heading { margin:1.45rem 0 .8rem; }
        .optimizer-section-heading span { color:var(--teal); font:700 .63rem 'DM Sans',sans-serif; letter-spacing:.17em; }
        .optimizer-section-heading h2 { margin:.22rem 0 0; color:var(--navy) !important; font:700 1.3rem 'Space Grotesk',sans-serif; }
        [class*="st-key-optimizer_lab_page"] [data-baseweb="input"] > div,
        [class*="st-key-optimizer_lab_page"] [data-baseweb="select"] > div,
        [class*="st-key-optimizer_lab_page"] [data-baseweb="multi-select"] > div { background:#F7FAFD !important; border:1.5px solid #AFC3D6 !important; border-radius:.68rem !important; }
        [class*="st-key-optimizer_lab_page"] [data-baseweb="input"] > div:focus-within,
        [class*="st-key-optimizer_lab_page"] [data-baseweb="select"] > div:focus-within,
        [class*="st-key-optimizer_lab_page"] [data-baseweb="multi-select"] > div:focus-within { background:#FFFFFF !important; border-color:var(--blue) !important; box-shadow:0 0 0 3px #4E73D826 !important; }
        [class*="st-key-optimizer_lab_page"] [data-testid="stMetric"] { min-height:5rem; }
        [class*="st-key-optimizer_lab_page"] [data-testid="stAlert"] { background:#FFFFFF; border:1px solid #C9D9E8; border-radius:.8rem; }
        [class*="st-key-optimizer_lab_page"] [data-testid="stCaptionContainer"] { color:var(--muted); }
        [class*="st-key-optimizer_run"] button { min-height:3.15rem !important; border:1px solid #2F5F83 !important; border-radius:.82rem !important; background:linear-gradient(115deg,#17263F,#2F5F83) !important; color:#FFFFFF !important; font:700 .98rem 'DM Sans',sans-serif !important; box-shadow:0 7px 18px #17263F24 !important; transition:transform .16s ease,box-shadow .16s ease,filter .16s ease; }
        [class*="st-key-optimizer_run"] button * { color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important; font:inherit !important; }
        [class*="st-key-optimizer_run"] button:hover { transform:translateY(-1px); filter:brightness(1.08); box-shadow:0 10px 22px #17263F30 !important; }
        [class*="st-key-optimizer_lab_page"] [data-testid="stTable"], [class*="st-key-optimizer_lab_page"] [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.8rem; overflow:hidden; }
        @media (max-width: 800px) {
            .block-container { padding:1.1rem 1rem 2rem; }
            .hero { padding:1.6rem; min-height:290px; }
            .hero-art { opacity:.32; right:0;width:58%; }
            .section-heading { align-items:start; gap:.75rem; flex-direction:column; }
            .feature-card { min-height:156px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
