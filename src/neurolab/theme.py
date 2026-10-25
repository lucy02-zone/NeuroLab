"""Shared visual system for NeuroLab's Streamlit pages."""

import streamlit as st


def apply_theme() -> None:
    """Apply a cleaner, consistent design system for the NeuroLab UI."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --page: #f3f6fa;
            --surface: #ffffff;
            --surface-soft: #eef5fb;
            --surface-strong: #e8f1f7;
            --line: #d7e2ec;
            --ink: #17263f;
            --muted: #61758b;
            --navy: #17263f;
            --blue: #4e73d8;
            --teal: #397f8b;
            --cyan: #49b7c4;
            --violet: #8b79ca;
            --shadow: rgba(23, 38, 63, 0.08);
        }

        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: var(--page) !important;
            color: var(--ink) !important;
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stHeader"] { background: transparent; box-shadow: none; }
        .block-container {
            max-width: 1440px;
            padding: 2rem 2rem 3rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--ink) !important;
            letter-spacing: -0.04em;
            margin: 0;
        }

        p, li, label, div, span { color: var(--ink); }
        p { color: var(--muted); line-height: 1.65; }
        a { color: var(--teal) !important; }

        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
            background: #f3f6fb !important;
            border-right: 1px solid #dfe8f3;
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] > div {
            padding: 1.2rem 0.9rem 1rem;
        }

        [data-testid="stSidebarCollapseButton"] button {
            display: grid !important;
            place-items: center;
            min-width: 2.35rem !important;
            min-height: 2.35rem !important;
            border: 1px solid #dfe7f2 !important;
            border-radius: 0.7rem !important;
            background: #ffffff !important;
            color: var(--navy) !important;
            box-shadow: 0 4px 12px rgba(23, 38, 63, 0.04);
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0.2rem 0 1.4rem;
            min-width: 0;
        }

        .brand-mark {
            flex: 0 0 2.8rem;
            display: grid;
            place-items: center;
            width: 2.8rem;
            height: 2.8rem;
            border: 1px solid rgba(78, 115, 216, 0.25);
            border-radius: 0.9rem;
            background: linear-gradient(145deg, #eef4ff 0%, #edfafc 100%);
            box-shadow: 0 8px 18px rgba(23, 38, 63, 0.05);
        }

        .brand-mark svg { width: 2rem; height: 2rem; }
        .brand-mark path { fill: none; stroke: #385e9d; stroke-width: 1.65; stroke-linecap: round; stroke-linejoin: round; }
        .brand-mark .node { stroke: #ffffff; stroke-width: 1.5; }
        .brand-mark .node-blue { fill: var(--blue); }
        .brand-mark .node-cyan { fill: var(--cyan); }
        .brand-mark .node-violet { fill: var(--violet); }

        .brand-name {
            color: var(--navy);
            font: 700 1.1rem 'Space Grotesk', sans-serif;
            letter-spacing: -0.03em;
        }

        .brand-tagline {
            margin-top: 0.25rem;
            color: var(--muted);
            font-size: 0.68rem;
            line-height: 1.4;
        }

        .sidebar-label {
            color: #7d8ea4;
            font-size: 0.64rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            margin: 1.5rem 0 0.7rem;
        }

        [data-testid="stSidebar"] .stButton {
            margin: 0.12rem 0;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            min-height: 2.7rem;
            justify-content: flex-start;
            text-align: left;
            padding: 0.55rem 0.72rem !important;
            border-radius: 0.72rem !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            color: var(--navy) !important;
            box-shadow: none !important;
            font: 600 0.88rem 'DM Sans', sans-serif !important;
        }

        [data-testid="stSidebar"] .stButton > button * {
            color: var(--navy) !important;
            -webkit-text-fill-color: var(--navy) !important;
        }

        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            background: #edf3fb !important;
            border-color: #d8e4f5 !important;
            color: var(--navy) !important;
        }

        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #edf5ff 0%, #eefaff 100%) !important;
            border-color: #cfe1ff !important;
            color: var(--navy) !important;
            box-shadow: inset 0 0 0 1px rgba(78, 115, 216, 0.08), 0 1px 0 rgba(23, 38, 63, 0.02) !important;
        }

        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {
            color: var(--navy) !important;
            -webkit-text-fill-color: var(--navy) !important;
        }

        .sidebar-footer {
            color: #72839a;
            font-size: 0.7rem;
            line-height: 1.5;
            margin-top: 1rem;
        }

        .eyebrow {
            color: var(--teal);
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .hero {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: space-between;
            overflow: hidden;
            min-height: 270px;
            padding: 2.3rem 2.5rem;
            border: 1px solid var(--line);
            border-radius: 1.3rem;
            background: linear-gradient(135deg, #ffffff 0%, #edf5fb 100%);
            box-shadow: 0 16px 38px rgba(23, 38, 63, 0.05);
            margin: 0.4rem 0 2rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 82% 50%, rgba(139, 121, 202, 0.14), transparent 42%);
            pointer-events: none;
        }

        .hero-copy {
            position: relative;
            z-index: 1;
            max-width: 630px;
        }

        .hero-kicker {
            color: var(--teal);
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0.55rem 0 0.25rem;
            font-size: clamp(2.7rem, 5vw, 4.4rem);
            line-height: 0.98;
        }

        .hero-period { color: var(--cyan); }
        .hero-tagline {
            margin: 0.3rem 0 0.6rem;
            color: var(--navy);
            font: 600 1rem 'Space Grotesk', sans-serif;
        }

        .hero-description {
            margin: 0.5rem 0 0.9rem;
            max-width: 560px;
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.7;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.8rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.8);
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 600;
        }

        .status-dot {
            display: inline-block;
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: var(--teal);
        }

        .hero-art {
            position: absolute;
            right: 4%;
            top: 12%;
            width: 34%;
            height: 70%;
            opacity: 0.8;
            background: linear-gradient(120deg, rgba(73, 183, 196, 0.12), rgba(139, 121, 202, 0.08));
            border: 1px solid rgba(121, 167, 255, 0.06);
            border-radius: 1.5rem;
            transform: rotate(-8deg);
        }

        .orbit {
            position: absolute;
            border: 1px solid rgba(57, 127, 139, 0.22);
            border-radius: 50%;
            transform: rotate(-20deg);
        }

        .orbit-one { width: 250px; height: 150px; top: 28px; left: 40px; }
        .orbit-two { width: 320px; height: 185px; top: 10px; left: 12px; }

        .neuron {
            position: absolute;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--blue);
            border: 2px solid #ffffff;
            box-shadow: 0 0 0 5px rgba(78, 115, 216, 0.12);
        }

        .n1 { left: 5%; top: 46%; }
        .n2 { left: 24%; top: 18%; background: var(--cyan); }
        .n3 { left: 24%; top: 72%; background: var(--cyan); }
        .n4 { left: 46%; top: 12%; }
        .n5 { left: 46%; top: 42%; background: var(--violet); }
        .n6 { left: 46%; top: 78%; }
        .n7 { left: 71%; top: 24%; background: var(--cyan); }
        .n8 { left: 71%; top: 66%; background: var(--cyan); }
        .n9 { left: 94%; top: 46%; }
        .n10 { left: 24%; top: 46%; width: 8px; height: 8px; }
        .n11 { left: 70%; top: 46%; width: 8px; height: 8px; background: var(--violet); }

        .section-heading,
        .dashboard-group-heading,
        .quiz-section-heading,
        .optimizer-section-heading {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            margin: 1.2rem 0 0.8rem;
        }

        .section-heading h2,
        .quiz-section-heading h2,
        .optimizer-section-heading h2 {
            font-size: 1.5rem;
            line-height: 1.2;
        }

        .section-note {
            color: var(--muted);
            font-size: 0.74rem;
            line-height: 1.5;
        }

        .dashboard-group-heading span,
        .quiz-section-heading span,
        .optimizer-section-heading span {
            color: var(--teal);
            font: 700 0.65rem 'DM Sans', sans-serif;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .dashboard-group-heading p {
            margin: 0.28rem 0 0;
            color: var(--muted);
            font-size: 0.78rem;
        }

        .feature-card,
        .info-panel,
        .placeholder-card,
        [data-testid="stMetricsContainer"] > div,
        [data-testid="stMetric"],
        [data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 1rem;
            box-shadow: 0 8px 20px rgba(23, 38, 63, 0.04);
        }

        .feature-card {
            min-height: 176px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }

        .feature-card:hover {
            transform: translateY(-2px);
            border-color: #b8cfe2;
            box-shadow: 0 12px 24px rgba(23, 38, 63, 0.08);
        }

        .feature-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
        }

        .feature-icon {
            display: grid;
            place-items: center;
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 0.75rem;
            background: var(--surface-strong);
            color: var(--blue);
            font-size: 1.02rem;
        }

        .blue .feature-icon { background: #eaf3ff; color: var(--blue); }
        .cyan .feature-icon { background: #e5f7f9; color: var(--teal); }
        .violet .feature-icon { background: #f0ebff; color: var(--violet); }

        .feature-number {
            color: #7d90a4;
            font: 600 0.68rem 'Space Grotesk', sans-serif;
        }

        .feature-card h3,
        .dashboard-module-card h3 {
            margin: 0 0 0.35rem;
            color: var(--navy) !important;
            font: 700 0.98rem 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
        }

        .feature-card p,
        .dashboard-module-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.55;
        }

        [class*="st-key-dashboard_module_"] {
            min-height: 12rem;
            border-radius: 1rem !important;
            background: #ffffff !important;
            border-color: var(--line) !important;
            box-shadow: 0 8px 20px rgba(23, 38, 63, 0.035);
        }

        .dashboard-module-card {
            min-height: 7.2rem;
            padding: 0.2rem 0;
        }

        [class*="st-key-dashboard_open_"] button {
            min-height: 2.5rem !important;
            border: 1px solid var(--line) !important;
            border-radius: 0.7rem !important;
            background: #f7fafd !important;
            color: var(--navy) !important;
            font: 700 0.78rem 'DM Sans', sans-serif !important;
        }

        .info-panel {
            min-height: 160px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 1rem;
        }

        .info-panel h2 {
            margin: 0.2rem 0 0.4rem;
            font-size: 1.1rem;
        }

        .progress-caption {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 0.9rem;
            color: var(--muted);
            font-size: 0.68rem;
        }

        .stack-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }

        .stack-list span {
            padding: 0.42rem 0.62rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: #f7fafc;
            color: var(--teal);
            font-size: 0.72rem;
            font-weight: 700;
        }

        .page-footer {
            margin-top: 1.5rem;
            text-align: center;
            color: var(--muted);
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .page-footer span {
            color: var(--teal);
            padding: 0 0.3rem;
        }

        .placeholder-card {
            max-width: 680px;
            margin: 2rem auto;
            padding: 3rem 2rem;
            text-align: center;
        }

        .placeholder-card h2 { margin: 0.8rem 0 0.5rem; }
        .placeholder-icon {
            display: grid;
            place-items: center;
            width: 3.2rem;
            height: 3.2rem;
            margin: 0 auto;
            border-radius: 1rem;
            background: var(--surface-soft);
            color: var(--blue);
            font-size: 1.5rem;
        }

        .stButton > button,
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-secondary"] {
            min-height: 2.75rem;
            border-radius: 0.75rem !important;
            font: 700 0.88rem 'DM Sans', sans-serif !important;
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .stButton > button,
        .stButton > button *,
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primary"] *,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-secondary"] * {
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
        }

        .stButton > button,
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, var(--navy) 0%, #2e547e 100%) !important;
            border: 1px solid var(--navy) !important;
            color: #ffffff !important;
            box-shadow: 0 8px 18px rgba(23, 38, 63, 0.12);
        }

        .stButton > button:hover,
        [data-testid="stBaseButton-primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(23, 38, 63, 0.16);
        }

        [data-testid="stBaseButton-secondary"] {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            color: var(--navy) !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {
            color: #ebf4ff !important;
            -webkit-text-fill-color: #ebf4ff !important;
        }

        [data-testid="stBaseButton-secondary"]:hover {
            border-color: #b3c4d6 !important;
            background: #f6fafd !important;
        }

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            border-radius: 0.7rem !important;
            box-shadow: none !important;
        }

        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within,
        [data-baseweb="textarea"] > div:focus-within {
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 3px rgba(78, 115, 216, 0.14) !important;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 0.85rem;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 18px rgba(23, 38, 63, 0.03);
        }

        [data-testid="stMetricLabel"] { color: var(--muted) !important; }
        [data-testid="stMetricValue"] {
            color: var(--navy) !important;
            font-family: 'Space Grotesk', sans-serif;
        }

        [data-testid="stTabs"] [role="tab"] {
            color: var(--muted) !important;
            font-weight: 600;
        }

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--navy) !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: var(--teal) !important;
        }

        [data-testid="stExpander"] {
            background: #ffffff;
            border-color: var(--line);
        }

        [data-testid="stAlert"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 0.8rem;
            color: var(--ink);
        }

        @media (max-width: 820px) {
            .block-container { padding: 1rem 0.9rem 2rem !important; }
            .hero { padding: 1.4rem; min-height: 240px; }
            .hero h1 { font-size: clamp(2.25rem, 9vw, 3.2rem); }
            .section-heading,
            .dashboard-group-heading,
            .quiz-section-heading,
            .optimizer-section-heading {
                flex-direction: column;
                align-items: flex-start;
            }
            .hero-art { opacity: 0.2; right: -8%; width: 58%; }
            [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.2rem; }
        }

        @media (max-width: 540px) {
            .hero { min-height: 220px; padding: 1.1rem; }
            .hero h1 { font-size: clamp(2.1rem, 11vw, 2.8rem); }
            .hero-description { font-size: 0.82rem; }
            .hero-kicker, .eyebrow { font-size: 0.58rem; letter-spacing: 0.12em; }
            .brand-mark { width: 2.4rem; height: 2.4rem; }
        }

        /* Keep the NeuroLab navigation legible and distinctive on deep navy. */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            background: #17263f !important;
            border-right: 1px solid #304763 !important;
            color: #ebf4ff !important;
        }
        [data-testid="stSidebar"] .brand-name { color: #ffffff !important; }
        [data-testid="stSidebar"] .brand-tagline { color: #bdcddd !important; }
        [data-testid="stSidebar"] .brand-mark {
            border-color: #7098e866 !important;
            background: linear-gradient(145deg, #233c5e, #315982) !important;
            box-shadow: 0 7px 18px #08152555 !important;
        }
        [data-testid="stSidebar"] .brand-mark path { stroke: #8db8ff !important; }
        [data-testid="stSidebar"] .brand-mark .node { stroke: #e6f0ff !important; }
        [data-testid="stSidebar"] .sidebar-label { color: #92a9c2 !important; }
        [data-testid="stSidebar"] .sidebar-footer { color: #91a5bc !important; }
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stButton > button * {
            color: #d5e0ec !important;
            -webkit-text-fill-color: #d5e0ec !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            border: 1px solid transparent !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            background: #213650 !important;
            border-color: #314b6b !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            background: #2a4566 !important;
            border-color: #426b91 !important;
            box-shadow: inset 3px 0 #49b7c4, 0 5px 14px #08152533 !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            border: 1px solid #49627e !important;
            background: #223752 !important;
            color: #f3f6fa !important;
        }
        [data-testid="stSidebarCollapseButton"] button svg,
        [data-testid="stSidebarCollapseButton"] button svg path {
            color: #f3f6fa !important;
            fill: #f3f6fa !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            display: grid !important;
            place-items: center !important;
            width: 2.4rem !important;
            height: 2.4rem !important;
            min-width: 2.4rem !important;
            min-height: 2.4rem !important;
            padding: 0 !important;
            border: 1px solid #536d8a !important;
            border-radius: 0.72rem !important;
            background: #2a4566 !important;
            color: #ffffff !important;
            box-shadow: 0 3px 9px #08152533 !important;
            transition: background-color 0.15s ease, border-color 0.15s ease !important;
        }
        [data-testid="stSidebarCollapseButton"] button svg { display: none !important; }
        [data-testid="stSidebarCollapseButton"] button::after {
            content: "×";
            display: block;
            color: #ffffff !important;
            font: 500 1.55rem/1 'DM Sans', sans-serif !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: #355575 !important;
            border-color: #7e9bb8 !important;
        }

        /* Dashboard launch buttons keep one readable color in every click state. */
        [class*="st-key-dashboard_open_"] .stButton > button,
        [class*="st-key-dashboard_open_"] .stButton > button:hover,
        [class*="st-key-dashboard_open_"] .stButton > button:focus,
        [class*="st-key-dashboard_open_"] .stButton > button:focus-visible,
        [class*="st-key-dashboard_open_"] .stButton > button:active {
            background: linear-gradient(135deg, #17263f 0%, #354d78 100%) !important;
            border: 1px solid #17263f !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
            transform: none !important;
            filter: none !important;
            box-shadow: none !important;
            transition: transform 0.16s ease !important;
        }
        [class*="st-key-dashboard_open_"] .stButton > button:hover { transform: translateY(-1px) !important; }
        [class*="st-key-dashboard_open_"] .stButton > button *,
        [class*="st-key-dashboard_open_"] .stButton > button:hover *,
        [class*="st-key-dashboard_open_"] .stButton > button:focus *,
        [class*="st-key-dashboard_open_"] .stButton > button:focus-visible *,
        [class*="st-key-dashboard_open_"] .stButton > button:active * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }

        /* Network Playground: make each step and its controls visibly defined. */
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] {
            border: 2px solid #b5c7d9 !important;
            border-radius: 0.95rem !important;
            box-shadow: 0 5px 16px rgba(23, 38, 63, 0.07) !important;
        }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] summary,
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] summary p {
            color: #17263f !important;
            font-weight: 700 !important;
        }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] summary p {
            font-family: 'Space Grotesk', sans-serif !important;
        }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [data-baseweb="input"] > div,
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [data-baseweb="select"] > div {
            background: #f7fafd !important;
            border: 1.7px solid #a8bdcf !important;
            border-radius: 0.68rem !important;
            box-shadow: 0 1px 3px rgba(23, 38, 63, 0.07) !important;
        }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [data-baseweb="input"] > div:focus-within,
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [data-baseweb="select"] > div:focus-within {
            background: #ffffff !important;
            border: 2px solid #4e73d8 !important;
            box-shadow: 0 0 0 3px rgba(78, 115, 216, 0.16) !important;
        }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] input,
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [data-baseweb="select"] div {
            color: #17263f !important;
            -webkit-text-fill-color: #17263f !important;
            font-weight: 600 !important;
        }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] label,
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
            color: #40566f !important;
            font-weight: 600 !important;
        }
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [aria-disabled="true"],
        [class*="st-key-network_playground_page"] [data-testid="stExpander"] [data-disabled="true"] {
            color: #71849a !important;
            -webkit-text-fill-color: #71849a !important;
            opacity: 1 !important;
        }

        /* Backpropagation Explorer: guided, readable panels matching the other labs. */
        [class*="st-key-backpropagation_explorer_page"] .backprop-hero {
            min-height: 245px;
            margin: 0.5rem 0 1.35rem;
        }
        .backprop-section-heading {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 1.45rem 0 0.75rem;
        }
        .backprop-section-heading > span {
            display: inline-grid;
            place-items: center;
            min-width: 2rem;
            height: 2rem;
            border: 1px solid #c9d9e8;
            border-radius: 0.65rem;
            background: #eaf1f7;
            color: #2f5f83;
            font: 700 0.7rem 'Space Grotesk', sans-serif;
        }
        .backprop-section-heading h2 {
            margin: 0;
            color: #17263f !important;
            font: 700 1.15rem 'Space Grotesk', sans-serif;
        }
        [class*="st-key-backprop_configuration_panel"],
        [class*="st-key-backprop_sample_panel"] {
            margin: 0 0 1rem;
            padding: 1rem 1.15rem 0.6rem;
            border: 2px solid #b5c7d9 !important;
            border-radius: 0.95rem !important;
            background: #ffffff !important;
            box-shadow: 0 5px 16px rgba(23, 38, 63, 0.06) !important;
        }
        [class*="st-key-backpropagation_explorer_page"] label,
        [class*="st-key-backpropagation_explorer_page"] [data-testid="stCaptionContainer"] {
            color: #40566f !important;
            font-weight: 600 !important;
        }
        [class*="st-key-backpropagation_explorer_page"] [data-baseweb="input"] > div,
        [class*="st-key-backpropagation_explorer_page"] [data-baseweb="select"] > div {
            background: #f7fafd !important;
            border: 1.7px solid #a8bdcf !important;
            border-radius: 0.68rem !important;
        }
        [class*="st-key-backpropagation_explorer_page"] [data-baseweb="input"] > div:focus-within,
        [class*="st-key-backpropagation_explorer_page"] [data-baseweb="select"] > div:focus-within {
            background: #ffffff !important;
            border: 2px solid #4e73d8 !important;
            box-shadow: 0 0 0 3px rgba(78, 115, 216, 0.15) !important;
        }
        [class*="st-key-backpropagation_explorer_page"] input,
        [class*="st-key-backpropagation_explorer_page"] [data-baseweb="select"] div {
            color: #17263f !important;
            -webkit-text-fill-color: #17263f !important;
            font-weight: 600 !important;
        }
        [class*="st-key-backpropagation_explorer_page"] [data-testid="stExpander"] {
            border: 1.5px solid #b5c7d9 !important;
            border-radius: 0.85rem !important;
            background: #ffffff !important;
        }
        [class*="st-key-backpropagation_explorer_page"] [data-testid="stExpander"] summary p {
            color: #17263f !important;
            font-weight: 700 !important;
        }
        [class*="st-key-backpropagation_explorer_page"] [data-testid="stMetric"] {
            min-height: 5rem;
            border: 1.5px solid #c4d3e1;
        }
        [class*="st-key-backpropagation_explorer_page"] [data-testid="stTable"],
        [class*="st-key-backpropagation_explorer_page"] [data-testid="stDataFrame"] {
            border: 1.5px solid #c4d3e1;
            border-radius: 0.8rem;
            overflow: hidden;
        }
        @media (max-width: 820px) {
            [class*="st-key-backprop_configuration_panel"],
            [class*="st-key-backprop_sample_panel"] { padding: 0.8rem 0.8rem 0.4rem; }
        }

        /* Learn DNN: distinct course header, progress controls, and lesson surface. */
        [class*="st-key-learn_dnn_page"] .learn-hero {
            min-height: 245px;
            margin: 0.5rem 0 1.2rem;
        }
        [class*="st-key-learn_dnn_controls_panel"],
        [class*="st-key-learn_dnn_lesson_panel"] {
            border: 2px solid #b5c7d9 !important;
            border-radius: 1rem !important;
            background: #ffffff !important;
            box-shadow: 0 5px 16px rgba(23, 38, 63, 0.06) !important;
        }
        [class*="st-key-learn_dnn_controls_panel"] {
            padding: 1rem 1.15rem 0.65rem;
            margin-bottom: 1rem;
        }
        [class*="st-key-learn_dnn_lesson_panel"] {
            padding: 1.25rem 1.4rem;
        }
        .learn-module-title {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding-bottom: 0.85rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid #d8e2ec;
        }
        .learn-module-title span {
            flex: 0 0 auto;
            padding: 0.38rem 0.6rem;
            border: 1px solid #c9d9e8;
            border-radius: 2rem;
            background: #eaf1f7;
            color: #2f5f83;
            font: 700 0.64rem 'DM Sans', sans-serif;
            letter-spacing: 0.1em;
        }
        .learn-module-title h2 {
            margin: 0;
            color: #17263f !important;
            font: 700 1.3rem 'Space Grotesk', sans-serif;
        }
        [class*="st-key-learn_dnn_page"] h4 {
            margin: 1.2rem 0 0.55rem;
            color: #17263f !important;
            font: 700 1rem 'Space Grotesk', sans-serif;
        }
        [class*="st-key-learn_dnn_page"] [data-testid="stExpander"] {
            margin: 0.55rem 0;
            border: 1.5px solid #c1d0df !important;
            border-radius: 0.8rem !important;
            background: #ffffff !important;
            box-shadow: 0 3px 10px rgba(23, 38, 63, 0.035);
        }
        [class*="st-key-learn_dnn_page"] [data-testid="stExpander"] summary p {
            color: #17263f !important;
            font-weight: 700 !important;
        }
        [class*="st-key-learn_dnn_page"] [data-testid="stExpander"] summary:hover p {
            color: #397f8b !important;
        }
        [class*="st-key-learn_dnn_page"] [data-baseweb="input"] > div,
        [class*="st-key-learn_dnn_page"] [data-baseweb="select"] > div {
            background: #f7fafd !important;
            border: 1.5px solid #a8bdcf !important;
            border-radius: 0.68rem !important;
        }
        [class*="st-key-learn_dnn_page"] [data-baseweb="input"] > div:focus-within,
        [class*="st-key-learn_dnn_page"] [data-baseweb="select"] > div:focus-within {
            border: 2px solid #4e73d8 !important;
            box-shadow: 0 0 0 3px rgba(78, 115, 216, 0.14) !important;
        }
        [class*="st-key-learn_dnn_page"] label,
        [class*="st-key-learn_dnn_page"] [data-testid="stCaptionContainer"] {
            color: #40566f !important;
        }
        [class*="st-key-learn_dnn_page"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 0.4rem; }
        [class*="st-key-learn_dnn_page"] [data-testid="stRadio"] div[role="radiogroup"] > label {
            min-height: 2.75rem;
            padding: 0.6rem 0.8rem;
            border: 1px solid #c9d9e8;
            border-radius: 0.72rem;
            background: #f8fbfd;
            color: #17263f !important;
        }
        [class*="st-key-learn_dnn_page"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
            border-color: #4e73d8;
            background: #edf3fc;
        }
        [class*="st-key-learn_dnn_page"] [data-testid="stMetric"] { min-height: 5rem; }
        [class*="st-key-learn_dnn_page"] [data-testid="stTable"],
        [class*="st-key-learn_dnn_page"] [data-testid="stDataFrame"] {
            border: 1px solid #c9d9e8;
            border-radius: 0.75rem;
            overflow: hidden;
        }
        [class*="st-key-learn_check_"] button,
        [class*="st-key-learn_link_"] button,
        [class*="st-key-learn_dnn_reset"] button {
            min-height: 2.8rem !important;
        }
        @media (max-width: 820px) {
            [class*="st-key-learn_dnn_lesson_panel"] { padding: 0.9rem; }
            [class*="st-key-learn_dnn_controls_panel"] { padding: 0.8rem 0.8rem 0.45rem; }
            .learn-module-title { align-items: flex-start; flex-direction: column; gap: 0.55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
