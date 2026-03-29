# =============================================================================
# VIT BHOPAL — AI Course BYOP: Social Media Recommendation Engine
# File: app.py
# Features: User accounts, post creation, AI feed ranking, like/disinterest
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, hashlib, time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="VIT Connect",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#                                   CSS HACKS
# =============================================================================
# Spent way too much time making this look good. 
# Overriding Streamlit's default UI to give it a modern "Glassmorphism" app feel. 
# If the AI doesn't impress the professor, hopefully the UI will! 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #07080f;
    --bg2:     #0c0e1a;
    --surface: #111422;
    --surf2:   #171a2e;
    --border:  #1e2240;
    --bord2:   #252a4a;
    --accent:  #4f8ef7;
    --acc2:    #7c5cfc;
    --green:   #22d3a0;
    --red:     #f75f7a;
    --yellow:  #fbbf24;
    --pink:    #f472b6;
    --text:    #e2e8ff;
    --muted:   #6b7499;
}

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bord2); border-radius: 3px; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Top bar ── */
.top-bar {
    width:100%; height:3px;
    background:linear-gradient(90deg,#4f8ef7,#7c5cfc,#f472b6,#22d3a0,#4f8ef7);
    background-size:300% 100%;
    animation:gradflow 4s linear infinite;
    margin-bottom:0;
}
@keyframes gradflow { to { background-position:300% 50%; } }

/* ── AUTH SCREENS ── */
.auth-wrap {
    max-width:420px; margin:60px auto; padding:40px;
    background:var(--surface); border:1px solid var(--border);
    border-radius:24px;
    box-shadow:0 24px 80px rgba(0,0,0,.5);
}
.auth-logo {
    font-size:36px; font-weight:900; text-align:center;
    background:linear-gradient(135deg,#4f8ef7,#7c5cfc,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:6px;
}
.auth-sub { text-align:center; color:var(--muted); font-size:14px; margin-bottom:28px; }
.auth-tab-wrap {
    display:flex; background:var(--bg2); border-radius:12px;
    padding:4px; margin-bottom:24px;
}
.auth-tab {
    flex:1; text-align:center; padding:8px;
    border-radius:10px; font-size:14px; font-weight:600;
    cursor:pointer; transition:all .2s; color:var(--muted);
}
.auth-tab.active { background:var(--surf2); color:var(--text); box-shadow:0 2px 8px rgba(0,0,0,.3); }

/* ── HERO ── */
.hero-title {
    font-size:38px; font-weight:900; letter-spacing:-1px;
    background:linear-gradient(135deg,#4f8ef7,#7c5cfc,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    padding-top:22px; line-height:1.1;
}
.hero-sub { font-size:14px; color:var(--muted); margin-top:5px; }
.live-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:linear-gradient(135deg,#4f8ef712,#7c5cfc12);
    border:1px solid #4f8ef755; border-radius:20px;
    padding:4px 14px; font-size:12px; color:#7cb3ff;
    font-family:'JetBrains Mono',monospace; margin-top:10px;
    animation:pulse-b 2s ease-in-out infinite;
}
@keyframes pulse-b { 0%,100%{border-color:#4f8ef755} 50%{border-color:#4f8ef7cc;box-shadow:0 0 14px #4f8ef733} }
.live-dot { width:7px;height:7px;border-radius:50%;background:#22d3a0;display:inline-block;animation:blink 1.5s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── STATS RIBBON ── */
.stats-ribbon { display:flex; gap:12px; margin:18px 0 20px; flex-wrap:wrap; }
.stat-pill {
    display:flex; align-items:center; gap:10px;
    background:var(--surface); border:1px solid var(--border);
    border-radius:14px; padding:12px 18px;
    transition:all .25s; cursor:default;
}
.stat-pill:hover { border-color:var(--accent); transform:translateY(-3px); box-shadow:0 10px 30px rgba(79,142,247,.18); }
.stat-pill .snum {
    font-size:21px; font-weight:800; font-family:'JetBrains Mono',monospace;
    background:linear-gradient(135deg,#4f8ef7,#7c5cfc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.stat-pill .slabel { font-size:11px; color:var(--muted); margin-top:1px; }

/* ── POST CARD ── */
.post-card {
    background:var(--surface); border-radius:18px;
    padding:22px 24px 16px; margin-bottom:6px;
    position:relative; overflow:hidden;
    transition:transform .22s, box-shadow .22s, border-color .22s;
    animation:fadeUp .35s ease both;
    border-left:3px solid var(--card-color,#4f8ef7);
    border-top:1px solid var(--border);
    border-right:1px solid var(--border);
    border-bottom:1px solid var(--border);
}
.post-card:hover { transform:translateY(-4px); box-shadow:0 16px 48px rgba(0,0,0,.5),0 0 0 1px var(--card-color,#4f8ef7)33; }
@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

.cc-clubs   { --card-color:#7c5cfc; }
.cc-exams   { --card-color:#f75f7a; }
.cc-canteen { --card-color:#22d3a0; }
.cc-events  { --card-color:#fbbf24; }
.cc-hostel  { --card-color:#29b6f6; }
.cc-general { --card-color:#4f8ef7; }

.mbadge {
    position:absolute; top:18px; right:18px;
    font-family:'JetBrains Mono',monospace; font-size:11px;
    padding:4px 12px; border-radius:20px;
    background:linear-gradient(135deg,#4f8ef712,#7c5cfc12);
    border:1px solid #4f8ef744; color:#7cb3ff;
}
.mbadge.hot { background:linear-gradient(135deg,#22d3a018,#4f8ef718); border-color:#22d3a066; color:#4de8c0; animation:glow-g 2s ease-in-out infinite; }
@keyframes glow-g { 0%,100%{box-shadow:0 0 0 #22d3a000} 50%{box-shadow:0 0 14px #22d3a055} }

.cbadge { display:inline-flex; align-items:center; gap:5px; font-size:10px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; padding:4px 12px; border-radius:20px; margin-bottom:10px; }
.cb-clubs   { background:#7c5cfc18; color:#a78fff; border:1px solid #7c5cfc44; }
.cb-exams   { background:#f75f7a18; color:#ff8fa0; border:1px solid #f75f7a44; }
.cb-canteen { background:#22d3a018; color:#4de8c0; border:1px solid #22d3a044; }
.cb-events  { background:#fbbf2418; color:#fcd56a; border:1px solid #fbbf2444; }
.cb-hostel  { background:#29b6f618; color:#60d0f8; border:1px solid #29b6f644; }
.cb-general { background:#4f8ef718; color:#7cb3ff; border:1px solid #4f8ef744; }

.post-title { font-size:17px; font-weight:700; color:var(--text); margin-bottom:8px; line-height:1.35; padding-right:110px; }
.post-body  { font-size:14px; color:var(--muted); line-height:1.7; margin-bottom:12px; }
.post-meta  { font-size:12px; color:#3d4560; display:flex; gap:14px; flex-wrap:wrap; margin-bottom:4px; }

/* ── AVATAR ── */
.avatar {
    width:38px; height:38px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; font-weight:800; flex-shrink:0;
    border:2px solid var(--border);
}

/* ── USER CHIP ── */
.user-chip {
    display:flex; align-items:center; gap:10px;
    background:var(--surf2); border:1px solid var(--border);
    border-radius:50px; padding:6px 14px 6px 6px;
    margin-bottom:16px;
}
.user-chip-name { font-size:14px; font-weight:600; }
.user-chip-role { font-size:11px; color:var(--muted); }

/* ── CREATE POST BOX ── */
.create-post-box {
    background:var(--surface); border:1px solid var(--border);
    border-radius:18px; padding:20px 22px;
    margin-bottom:20px;
    transition:border-color .2s;
}
.create-post-box:hover { border-color:var(--accent); }
.create-post-label { font-size:13px; font-weight:600; color:var(--muted); margin-bottom:12px; }

/* ── SIDEBAR ── */
.sb-brand { background:linear-gradient(135deg,var(--surface),var(--surf2)); border-bottom:1px solid var(--border); padding:20px 18px 14px; }
.sb-brand-name { font-size:22px; font-weight:800; background:linear-gradient(135deg,#4f8ef7,#7c5cfc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sb-brand-tag  { font-size:11px; color:var(--muted); font-family:'JetBrains Mono',monospace; margin-top:2px; }
.sb-sec { font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); margin:16px 0 8px 2px; }
.sb-metric { background:var(--surf2); border:1px solid var(--border); border-radius:12px; padding:12px 14px; display:flex; align-items:center; gap:12px; margin-bottom:8px; transition:all .2s; }
.sb-metric:hover { border-color:var(--accent); transform:translateX(3px); }
.sb-metric-val   { font-size:22px; font-weight:800; font-family:'JetBrains Mono',monospace; }
.sb-metric-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.8px; }
.int-cloud { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.int-tag { font-size:11px; padding:3px 11px; border-radius:20px; background:linear-gradient(135deg,#4f8ef712,#7c5cfc12); border:1px solid #4f8ef733; color:#7cb3ff; font-family:'JetBrains Mono',monospace; }

/* ── PROFILE CARD ── */
.profile-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:18px; padding:24px; margin-bottom:16px;
    text-align:center;
}
.profile-avatar {
    width:72px; height:72px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:28px; font-weight:900; margin:0 auto 12px;
    border:3px solid transparent;
    background:linear-gradient(var(--surface),var(--surface)) padding-box,
               linear-gradient(135deg,#4f8ef7,#7c5cfc,#f472b6) border-box;
}
.profile-name { font-size:20px; font-weight:800; }
.profile-info { font-size:13px; color:var(--muted); margin-top:4px; }
.profile-stats { display:flex; justify-content:center; gap:28px; margin-top:16px; padding-top:16px; border-top:1px solid var(--border); }
.pstat-val   { font-size:22px; font-weight:800; font-family:'JetBrains Mono',monospace; background:linear-gradient(135deg,#4f8ef7,#7c5cfc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.pstat-label { font-size:11px; color:var(--muted); }

/* ── TABS ── */
.nav-tabs { display:flex; gap:6px; margin:18px 0 20px; }
.nav-tab {
    padding:8px 20px; border-radius:30px; font-size:13px;
    font-weight:600; border:1.5px solid var(--border);
    background:var(--surface); color:var(--muted);
    cursor:pointer; transition:all .2s; display:inline-flex; align-items:center; gap:6px;
}
.nav-tab.active {
    background:linear-gradient(135deg,#4f8ef722,#7c5cfc22);
    border-color:#4f8ef7; color:#7cb3ff;
    box-shadow:0 0 18px rgba(79,142,247,.2);
}

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    font-family:'Outfit',sans-serif !important;
    font-weight:600 !important; font-size:13px !important;
    border-radius:10px !important; padding:6px 18px !important;
    transition:all .2s !important;
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background:var(--surf2) !important;
    border:1px solid var(--bord2) !important;
    border-radius:12px !important;
    color:var(--text) !important;
    font-family:'Outfit',sans-serif !important;
}
.stTextInput > label, .stTextArea > label, .stSelectbox > label {
    color:var(--muted) !important; font-size:13px !important;
}
div[data-testid="stSelectbox"] > div > div { background:var(--surf2) !important; border:1px solid var(--bord2) !important; border-radius:12px !important; }
.stSuccess { background:linear-gradient(135deg,#22d3a018,#4f8ef718) !important; border:1px solid #22d3a055 !important; border-radius:12px !important; }
.stError   { background:linear-gradient(135deg,#f75f7a18,transparent) !important; border:1px solid #f75f7a55 !important; border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# Hardcoding some initial dummy data so the app isn't an empty ghost town 
#  Added Some relatable campus stuff (Maggi stalls, hackathons, hot water issues) to make it feel like the real VIT.
SEED_POSTS = [
    {"id":0,  "title":"Coding Club Hackathon This Weekend!","body":"Join us for a 24-hour hackathon focused on AI and machine learning projects. Open to all branches. Prizes worth ₹50,000!","category":"Clubs","author":"TechXcel Club","likes":142,"timestamp":"2h ago","emoji":"💻","user":"vitadmin"},
    {"id":1,  "title":"Photography Club: Golden Hour Walk","body":"Bring your cameras and phones. We walk the campus at sunset to capture architecture and nature. Beginners welcome!","category":"Clubs","author":"PixelVIT Club","likes":87,"timestamp":"4h ago","emoji":"📸","user":"vitadmin"},
    {"id":2,  "title":"Robotics Club Recruitment Open","body":"We are building an autonomous rover for the national competition. Looking for members with interest in embedded systems, CAD, and AI.","category":"Clubs","author":"RoboVIT","likes":203,"timestamp":"6h ago","emoji":"🤖","user":"vitadmin"},
    {"id":3,  "title":"Literary Club: Creative Writing Workshop","body":"Explore short fiction, poetry, and screenwriting in a free Saturday workshop. Get feedback from published authors.","category":"Clubs","author":"VIT Quill","likes":55,"timestamp":"1d ago","emoji":"✍️","user":"vitadmin"},
    {"id":4,  "title":"Music Club Open Mic Night — Register Now","body":"Singers, guitarists, beatboxers — the stage is yours. Open mic at VIT Amphitheatre this Friday, 6 PM.","category":"Clubs","author":"Resonance Music Club","likes":175,"timestamp":"1d ago","emoji":"🎵","user":"vitadmin"},
    {"id":5,  "title":"CAT 2025 Preparation Study Group Forming","body":"Looking for serious CAT aspirants to form a study group. Will cover quant, verbal, and DI sections together.","category":"Exams","author":"MBA Aspirants VIT","likes":94,"timestamp":"3h ago","emoji":"📊","user":"vitadmin"},
    {"id":6,  "title":"GATE CS Tips: 98 Percentile Resources","body":"Sharing my complete GATE CS study plan, notes, and YouTube playlists. DM for the Google Drive link.","category":"Exams","author":"Priya K.","likes":512,"timestamp":"5h ago","emoji":"🏆","user":"vitadmin"},
    {"id":7,  "title":"Mid-Semester Exam Schedule Released","body":"The official mid-semester exam timetable has been posted on the ERP portal. Check your hall ticket and room number.","category":"Exams","author":"Academic Office","likes":310,"timestamp":"2h ago","emoji":"📅","user":"vitadmin"},
    {"id":8,  "title":"Data Structures Lab Internal Assessment Tips","body":"Faculty confirmed the internal will cover linked lists, trees, and graphs. Practice LeetCode easy-medium problems.","category":"Exams","author":"CS Study Circle","likes":188,"timestamp":"7h ago","emoji":"🌳","user":"vitadmin"},
    {"id":9,  "title":"Mathematics III Formula Sheet — Shared","body":"Compiled a concise formula sheet for Transforms, PDE, and Statistics. Useful for the upcoming unit test.","category":"Exams","author":"Arjun M.","likes":270,"timestamp":"1d ago","emoji":"📐","user":"vitadmin"},
    {"id":10, "title":"New South Indian Corner in Block-C Canteen","body":"Freshly made dosas, idlis, and sambar now available from 7 AM. The filter coffee is absolutely worth it!","category":"Canteen","author":"Food Explorers VIT","likes":432,"timestamp":"1h ago","emoji":"🥞","user":"vitadmin"},
    {"id":11, "title":"Best Budget Meals Under ₹60 on Campus","body":"Ranked the cheapest filling meals: rajma-rice at Main Mess (₹40), egg roll near Gate 2 (₹50), maggi stall (₹30).","category":"Canteen","author":"Broke Foodie","likes":667,"timestamp":"3h ago","emoji":"🍱","user":"vitadmin"},
    {"id":12, "title":"Late Night Snacks: Stalls Open After 10 PM","body":"For those coding sessions that stretch past midnight — the chai stall near Hostel Block 7 now stays open till 1 AM.","category":"Canteen","author":"Night Owls VIT","likes":389,"timestamp":"5h ago","emoji":"🌙","user":"vitadmin"},
    {"id":13, "title":"Petition: Add Vegan Options to Main Mess","body":"We have 200 signatures. If you support adding a dedicated vegan food station, sign and share before Friday.","category":"Canteen","author":"VIT Green Society","likes":118,"timestamp":"2d ago","emoji":"🌱","user":"vitadmin"},
    {"id":14, "title":"Food Fest 2025: Street Food from 15 States","body":"The annual Food Fest is back! Stalls from all over India, cooking competitions, and a celebrity chef demo.","category":"Canteen","author":"Cultural Committee","likes":540,"timestamp":"6h ago","emoji":"🎪","user":"vitadmin"},
    {"id":15, "title":"IEEE Student Chapter: AI/ML Guest Lecture Series","body":"Industry experts from Google and Microsoft visit campus next week for a three-day lecture series on applied AI.","category":"Events","author":"IEEE VIT Bhopal","likes":298,"timestamp":"4h ago","emoji":"🎤","user":"vitadmin"},
    {"id":16, "title":"Sports Day: Registrations Close Tomorrow","body":"Events include cricket, football, badminton, chess, and athletics. Register on ERP. Free energy drinks for participants!","category":"Events","author":"Sports Council","likes":221,"timestamp":"8h ago","emoji":"🏅","user":"vitadmin"},
    {"id":17, "title":"Hostel Maintenance: Hot Water Schedule Changed","body":"Hot water is now available 5:30–7:30 AM and 8–10 PM due to the solar heater upgrade. Plan accordingly.","category":"Hostel","author":"Hostel Admin","likes":156,"timestamp":"9h ago","emoji":"🚿","user":"vitadmin"},
    {"id":18, "title":"Room Swap Forum — Post Your Requests Here","body":"Looking for a room swap to be closer to your branch block? Drop your current room and requirement in the comments.","category":"Hostel","author":"Hostel Council","likes":73,"timestamp":"1d ago","emoji":"🔄","user":"vitadmin"},
    {"id":19, "title":"Study Room in Block-A Now Open 24/7","body":"The air-conditioned study room in Block-A hostel is now open round the clock. Bring your ID card for entry.","category":"Hostel","author":"Hostel Admin","likes":345,"timestamp":"10h ago","emoji":"📚","user":"vitadmin"},
]

CAT_CONFIG = {
    "Clubs":   {"icon":"🎭","card_class":"cc-clubs",  "badge_class":"cb-clubs"},
    "Exams":   {"icon":"📝","card_class":"cc-exams",  "badge_class":"cb-exams"},
    "Canteen": {"icon":"🍽️","card_class":"cc-canteen","badge_class":"cb-canteen"},
    "Events":  {"icon":"🎉","card_class":"cc-events", "badge_class":"cb-events"},
    "Hostel":  {"icon":"🏠","card_class":"cc-hostel", "badge_class":"cb-hostel"},
    "General": {"icon":"💬","card_class":"cc-general","badge_class":"cb-general"},
}
CAT_ICONS = {"All":"✨","Clubs":"🎭","Exams":"📝","Canteen":"🍽️","Events":"🎉","Hostel":"🏠","General":"💬"}
AVATAR_COLORS = ["#7c5cfc","#4f8ef7","#22d3a0","#f472b6","#fbbf24","#f75f7a","#29b6f6"]
CAT_EMOJIS = {"Clubs":"🎭","Exams":"📝","Canteen":"🍽️","Events":"🎉","Hostel":"🏠","General":"💬"}


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def make_avatar_color(username):
    return AVATAR_COLORS[sum(ord(c) for c in username) % len(AVATAR_COLORS)]

def get_initials(name):
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()

def time_ago(ts):
    diff = int(time.time()) - int(ts)
    if diff < 60:    return "just now"
    if diff < 3600:  return f"{diff//60}m ago"
    if diff < 86400: return f"{diff//3600}h ago"
    return f"{diff//86400}d ago"

# Using Streamlit's session_state as an in-memory "Database". 
# Setting up Firebase or SQL was overkill for a BYOP prototype and would make the setup too complex for a quick demo. 
# Note: Data wipes on server restart, but works perfectly for the evaluation!
def init_state():
    if "users_db" not in st.session_state:
        st.session_state.users_db = {
            "vitadmin": {"password": hash_pw("admin123"), "name": "VIT Admin",
                         "branch": "Administration", "year": "Staff", "bio": "Official VIT Bhopal account"}
        }
    if "posts_db" not in st.session_state:
        st.session_state.posts_db = SEED_POSTS.copy()
        st.session_state.next_id  = len(SEED_POSTS)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.current_user = None
    if "agents" not in st.session_state:
        st.session_state.agents = {}      
    if "liked_posts" not in st.session_state:
        st.session_state.liked_posts = {} 
    if "disliked_posts" not in st.session_state:
        st.session_state.disliked_posts = {}
    if "fv" not in st.session_state:
        st.session_state.fv = 0
    if "page" not in st.session_state:
        st.session_state.page = "feed"
    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "login"

init_state()
  


def build_tfidf(posts):
    # --- CO2: Perception & Knowledge Representation ---
    # We can't feed raw English to the math model. 
    # Smashing the title, body, and category together to extract keywords.
    texts = [p["title"] + " " + p["body"] + " " + p["category"] for p in posts]
    if not texts:
        return None, None
    
    # Removing stop words (the, is, at) and looking at 1-2 word phrases.
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    mat = vec.fit_transform(texts)
    return vec, mat


class RecommendationAgent:
    # --- CO1: The Rational Agent ---
    def __init__(self, n):
        # User starts as a blank slate (vector of zeros)
        self.user_profile = np.zeros(n)
        self.scores       = {}

    def like_post(self, idx, vec):
        # --- CO4: Machine Learning (Adaptation) ---
        # When a user likes a post, we literally add the post's TF-IDF vector 
        # to the user's profile. The agent is learning their personality in real-time!
        self.user_profile += vec.toarray().flatten()

    def rank(self, mat, ids):
        # --- CO3: Probability & Reasoning ---
        # Cold start problem: If they haven't liked anything, return 0 score.
        if np.linalg.norm(self.user_profile) == 0:
            return {pid:0.0 for pid in ids}
        
        # Calculating Cosine Similarity between what the user likes and what the post is.
        # This acts as our predictive probability of engagement.
        sims = cosine_similarity(self.user_profile.reshape(1,-1), mat)[0]
        return {pid: float(sims[i]) for i, pid in enumerate(ids)}


def get_agent(username, n):
    if username not in st.session_state.agents:
        st.session_state.agents[username] = RecommendationAgent(n)
    agent = st.session_state.agents[username]
    if agent.user_profile.shape[0] != n:
        agent.user_profile = np.zeros(n)
    return agent

def get_liked(username):
    if username not in st.session_state.liked_posts:
        st.session_state.liked_posts[username] = set()
    return st.session_state.liked_posts[username]

def get_disliked(username):
    if username not in st.session_state.disliked_posts:
        st.session_state.disliked_posts[username] = set()
    return st.session_state.disliked_posts[username]


def show_auth():
    st.markdown("<div class='top-bar'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1,1.6,1])
    with col:
        st.markdown("""
        <div class='auth-wrap'>
            <div class='auth-logo'>VIT Connect 🎓</div>
            <div class='auth-sub'>Your campus social feed, powered by AI</div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Login", "🚀 Create Account"])

        with tab_login:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            uname = st.text_input("Username", key="li_user", placeholder="Enter your username")
            pw    = st.text_input("Password", type="password", key="li_pw", placeholder="Enter your password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Login →", use_container_width=True, key="login_btn"):
                db = st.session_state.users_db
                if uname in db and db[uname]["password"] == hash_pw(pw):
                    st.session_state.logged_in    = True
                    st.session_state.current_user = uname
                    st.session_state.page         = "feed"
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password.")
            st.markdown("""
            <div style='text-align:center;margin-top:14px;font-size:12px;color:#404870'>
                Demo: username <b style='color:#7cb3ff'>vitadmin</b> / password <b style='color:#7cb3ff'>admin123</b>
            </div>""", unsafe_allow_html=True)

        with tab_signup:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                new_fname = st.text_input("First Name", key="su_fn", placeholder="Arjun")
            with c2:
                new_lname = st.text_input("Last Name", key="su_ln", placeholder="Sharma")
            new_user  = st.text_input("Username", key="su_user", placeholder="arjun_sharma")
            new_branch= st.selectbox("Branch", ["CSE","ECE","ME","CE","EEE","IT","AIDS","Other"], key="su_branch")
            new_year  = st.selectbox("Year", ["1st Year","2nd Year","3rd Year","4th Year"], key="su_year")
            new_bio   = st.text_input("Bio (optional)", key="su_bio", placeholder="CSE student | AI enthusiast")
            new_pw    = st.text_input("Password", type="password", key="su_pw", placeholder="Create a password")
            new_pw2   = st.text_input("Confirm Password", type="password", key="su_pw2", placeholder="Repeat password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("Create Account 🚀", use_container_width=True, key="signup_btn"):
                if not all([new_fname, new_lname, new_user, new_pw]):
                    st.error("Please fill in all required fields.")
                elif new_user in st.session_state.users_db:
                    st.error("Username already taken. Try another.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    st.session_state.users_db[new_user] = {
                        "password": hash_pw(new_pw),
                        "name":     f"{new_fname} {new_lname}",
                        "branch":   new_branch,
                        "year":     new_year,
                        "bio":      new_bio or f"{new_branch} student at VIT Bhopal",
                    }
                    st.session_state.logged_in    = True
                    st.session_state.current_user = new_user
                    st.session_state.page         = "feed"
                    st.success("Account created! Welcome to VIT Connect 🎉")
                    time.sleep(0.8)
                    st.rerun()


def show_sidebar():
    username = st.session_state.current_user
    user     = st.session_state.users_db[username]
    liked    = get_liked(username)
    disliked = get_disliked(username)
    acolor   = make_avatar_color(username)
    initials = get_initials(user["name"])

    with st.sidebar:
        st.markdown("""
        <div class='sb-brand'>
            <div class='sb-brand-name'>VIT Connect</div>
            <div class='sb-brand-tag'>AI-Powered Feed · VIT Bhopal</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='padding:14px 18px 0'>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='user-chip'>
            <div class='avatar' style='background:linear-gradient(135deg,{acolor}44,{acolor}22);border-color:{acolor}55;color:{acolor}'>{initials}</div>
            <div>
                <div class='user-chip-name'>{user["name"]}</div>
                <div class='user-chip-role'>{user["branch"]} · {user["year"]}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        pages = [("🏠","Feed","feed"),("✏️","Create Post","create"),("👤","My Profile","profile")]
        for icon, label, pg in pages:
            active_style = "background:linear-gradient(135deg,#4f8ef722,#7c5cfc22);border-color:#4f8ef7;color:#7cb3ff;" if st.session_state.page == pg else ""
            if st.button(f"{icon}  {label}", key=f"nav_{pg}", use_container_width=True):
                st.session_state.page = pg
                st.rerun()

        st.markdown("---")
        st.markdown("<div class='sb-sec'>Your Activity</div>", unsafe_allow_html=True)

        my_posts = [p for p in st.session_state.posts_db if p.get("user") == username]
        st.markdown(f"""
        <div class='sb-metric'>
            <span style='font-size:20px'>📝</span>
            <div><div class='sb-metric-val' style='color:#7cb3ff'>{len(my_posts)}</div><div class='sb-metric-label'>My Posts</div></div>
        </div>
        <div class='sb-metric'>
            <span style='font-size:20px'>❤️</span>
            <div><div class='sb-metric-val' style='color:#f472b6'>{len(liked)}</div><div class='sb-metric-label'>Posts Liked</div></div>
        </div>
        <div class='sb-metric'>
            <span style='font-size:20px'>🚫</span>
            <div><div class='sb-metric-val' style='color:#f75f7a'>{len(disliked)}</div><div class='sb-metric-label'>Posts Hidden</div></div>
        </div>""", unsafe_allow_html=True)

        vec, mat = build_tfidf(st.session_state.posts_db)
        if vec and mat is not None:
            agent = get_agent(username, mat.shape[1])
            if np.linalg.norm(agent.user_profile) > 0:
                st.markdown("<div class='sb-sec'>Learned Interests</div>", unsafe_allow_html=True)
                fn   = vec.get_feature_names_out()
                idxs = agent.user_profile.argsort()[::-1][:12]
                kws  = [fn[i] for i in idxs if agent.user_profile[i] > 0]
                html = "<div class='int-cloud'>" + "".join(f"<span class='int-tag'>{k}</span>" for k in kws) + "</div>"
                st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in    = False
            st.session_state.current_user = None
            st.session_state.page         = "feed"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def show_feed():
    username = st.session_state.current_user
    liked    = get_liked(username)
    disliked = get_disliked(username)
    posts    = [p for p in st.session_state.posts_db if p["id"] not in disliked]

    st.markdown("<div class='top-bar'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-title'>VIT Connect 🎓</div>
    <div class='hero-sub'>Your personalised campus social feed, powered by AI</div>
    <div class='live-badge'><span class='live-dot'></span>Content-Based Filtering · TF-IDF + Cosine Similarity · Live</div>
    """, unsafe_allow_html=True)

    top_score = 0.0
    vec, mat  = build_tfidf(st.session_state.posts_db)
    if vec and mat is not None:
        agent     = get_agent(username, mat.shape[1])
        post_ids  = [p["id"] for p in posts]
        scores    = agent.rank(mat, list(range(len(st.session_state.posts_db))))
        vis_scores = {p["id"]: scores.get(p["id"],0.0) for p in posts}
        top_score  = max(vis_scores.values())*100 if vis_scores else 0

    st.markdown(f"""
    <div class='stats-ribbon'>
        <div class='stat-pill'><span style='font-size:20px'>📰</span><div><div class='snum'>{len(posts)}</div><div class='slabel'>Posts in Feed</div></div></div>
        <div class='stat-pill'><span style='font-size:20px'>❤️</span><div><div class='snum'>{len(liked)}</div><div class='slabel'>Liked</div></div></div>
        <div class='stat-pill'><span style='font-size:20px'>🧠</span><div><div class='snum'>{min(len(liked)*20,100)}%</div><div class='slabel'>Profile Strength</div></div></div>
        <div class='stat-pill'><span style='font-size:20px'>🎯</span><div><div class='snum'>{top_score:.0f}%</div><div class='slabel'>Top Match</div></div></div>
        <div class='stat-pill'><span style='font-size:20px'>🌐</span><div><div class='snum'>{len(st.session_state.posts_db)}</div><div class='slabel'>Total Posts</div></div></div>
    </div>""", unsafe_allow_html=True)

    fc, sc = st.columns([3,1])
    with fc:
        active_cat = st.selectbox("Category", ["All","Clubs","Exams","Canteen","Events","Hostel","General"],
                                  format_func=lambda x: f"{CAT_ICONS[x]}  {x}", label_visibility="collapsed")
    with sc:
        sort_mode = st.selectbox("Sort", ["🎯 Relevance","❤️ Most Liked","🕐 Recent"],
                                 label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    df = pd.DataFrame(posts)
    if vec and mat is not None:
        agent  = get_agent(username, mat.shape[1])
        all_scores = agent.rank(mat, list(range(len(st.session_state.posts_db))))

        # Mapping the ML relevance scores back to the dataframe
        df["relevance"] = df["id"].map(lambda i: all_scores.get(i,0.0))

        # Sorting logic: The "Smart Feed" kicks in when "Relevance" is selected.
        if "Relevance" in sort_mode:
            df = df.sort_values("relevance", ascending=False)
        elif "Most Liked" in sort_mode:
            df = df.sort_values("likes", ascending=False)
        elif "Recent" in sort_mode:
            df = df.sort_values("id", ascending=False)
    else:
        df["relevance"] = 0.0

    if active_cat != "All":
        df = df[df["category"] == active_cat]
    df = df.reset_index(drop=True)

    if df.empty:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:var(--muted)'>
            <div style='font-size:48px;margin-bottom:16px'>🌌</div>
            <div style='font-size:18px;color:var(--text)'>No posts here</div>
        </div>""", unsafe_allow_html=True)
        return

    for idx, row in df.iterrows():
        pid      = int(row["id"])
        score    = float(row.get("relevance", 0.0))
        cfg      = CAT_CONFIG.get(row["category"], CAT_CONFIG["General"])
        is_liked = pid in liked
        pct      = score * 100
        is_mine  = row.get("user","") == username

        acolor   = make_avatar_color(row.get("user","vitadmin"))
        post_user_info = st.session_state.users_db.get(row.get("user","vitadmin"), {})
        initials = get_initials(post_user_info.get("name", row.get("author","VIT")))

        if pct >= 60:
            badge_html = f"<span class='mbadge hot'>🔥 {pct:.0f}% match</span>"
        elif pct > 0:
            badge_html = f"<span class='mbadge'>⚡ {pct:.0f}% match</span>"
        else:
            badge_html = "<span class='mbadge'>✨ New</span>"

        heart = "❤️ " if is_liked else ""
        mine_tag = " <span style='font-size:10px;background:#4f8ef722;color:#7cb3ff;border:1px solid #4f8ef744;border-radius:10px;padding:2px 8px;margin-left:4px'>Your post</span>" if is_mine else ""

        ts = row.get("timestamp","")
        try:
            ts = time_ago(float(ts))
        except:
            pass

        st.markdown(f"""
<div class='post-card {cfg["card_class"]}' style='animation-delay:{idx*0.03}s'>
    {badge_html}
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>
        <div class='avatar' style='background:linear-gradient(135deg,{acolor}44,{acolor}22);border-color:{acolor}55;color:{acolor};width:36px;height:36px;font-size:14px'>{initials}</div>
        <div>
            <div style='font-size:13px;font-weight:600;color:var(--text)'>{post_user_info.get("name", row.get("author","VIT"))}{mine_tag}</div>
            <div style='font-size:11px;color:var(--muted)'>{ts}</div>
        </div>
    </div>
    <div class='cbadge {cfg["badge_class"]}'>{cfg["icon"]} {row["category"]}</div>
    <div class='post-title'>{heart}{row.get("emoji","💬")} {row["title"]}</div>
    <div class='post-body'>{row["body"]}</div>
    <div class='post-meta'>
        <span>❤️ {int(row["likes"]):,} likes</span>
    </div>
</div>
""", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns([1.1, 1.8, 1.5, 4])
        with b1:
            if is_liked:
                st.markdown("""<div style='background:linear-gradient(135deg,#f472b618,#f75f7a18);border:1px solid #f472b655;border-radius:10px;padding:7px 14px;font-size:13px;font-weight:600;color:#f472b6;text-align:center'>❤️ Liked</div>""", unsafe_allow_html=True)
            else:
                if st.button("🤍 Like", key=f"L{pid}_{st.session_state.fv}"):
                    liked.add(pid)
                    for p in st.session_state.posts_db:
                        if p["id"] == pid:
                            p["likes"] = p.get("likes",0) + 1
                    if vec and mat is not None:
                        agent = get_agent(username, mat.shape[1])
                        post_idx = next((i for i,p in enumerate(st.session_state.posts_db) if p["id"]==pid), None)
                        if post_idx is not None:
                            agent.like_post(post_idx, mat[post_idx])
                    st.session_state.fv += 1
                    st.rerun()
        with b2:
            if st.button("👎 Not Interested", key=f"D{pid}_{st.session_state.fv}"):
                disliked.add(pid)
                st.session_state.fv += 1
                st.rerun()
        with b3:
            if is_mine:
                if st.button("🗑️ Delete", key=f"DEL{pid}_{st.session_state.fv}"):
                    st.session_state.posts_db = [p for p in st.session_state.posts_db if p["id"] != pid]
                    st.session_state.fv += 1
                    st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def show_create_post():
    username = st.session_state.current_user
    user     = st.session_state.users_db[username]
    acolor   = make_avatar_color(username)
    initials = get_initials(user["name"])

    st.markdown("<div class='top-bar'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding-top:24px;margin-bottom:20px'>
        <div style='font-size:28px;font-weight:900;background:linear-gradient(135deg,#4f8ef7,#7c5cfc);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>Create Post ✏️</div>
        <div style='font-size:14px;color:var(--muted);margin-top:4px'>Share something with the VIT Bhopal community</div>
    </div>""", unsafe_allow_html=True)

    _, main, _ = st.columns([0.3, 3, 0.3])
    with main:
        st.markdown(f"""
        <div class='create-post-box'>
            <div style='display:flex;align-items:center;gap:12px;margin-bottom:18px'>
                <div class='avatar' style='background:linear-gradient(135deg,{acolor}44,{acolor}22);border-color:{acolor}55;color:{acolor}'>{initials}</div>
                <div>
                    <div style='font-weight:700;font-size:15px'>{user["name"]}</div>
                    <div style='font-size:12px;color:var(--muted)'>{user["branch"]} · {user["year"]}</div>
                </div>
            </div>
            <div style='color:var(--muted);font-size:13px;margin-bottom:4px'>What's on your mind?</div>
        </div>""", unsafe_allow_html=True)

        post_title    = st.text_input("Post Title *", placeholder="e.g. Study group for End-Sem forming!", key="np_title")
        post_body     = st.text_area("Post Content *", placeholder="Write your post here... share news, ask questions, or announce events.", height=140, key="np_body")

        c1, c2 = st.columns(2)
        with c1:
            post_cat  = st.selectbox("Category *", ["Clubs","Exams","Canteen","Events","Hostel","General"], key="np_cat")
        with c2:
            post_emoji = st.selectbox("Post Emoji", ["💬","📢","🎉","📝","🍕","🏆","💡","🤝","📸","🎵","🚀","⚡","🌟","🔥","❓"], key="np_emoji")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if post_title or post_body:
            cfg = CAT_CONFIG.get(post_cat, CAT_CONFIG["General"])
            st.markdown(f"""
            <div style='margin-bottom:16px'>
                <div style='font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Preview</div>
                <div class='post-card {cfg["card_class"]}'>
                    <div class='cbadge {cfg["badge_class"]}'>{cfg["icon"]} {post_cat}</div>
                    <div class='post-title'>{post_emoji} {post_title or "Your title here..."}</div>
                    <div class='post-body'>{post_body or "Your post content will appear here..."}</div>
                    <div class='post-meta'><span>✍️ {user["name"]}</span><span>🕐 just now</span><span>👍 0 likes</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

        pub_col, can_col = st.columns([1,1])
        with pub_col:
            if st.button("🚀 Publish Post", use_container_width=True, key="publish_btn"):
                if not post_title.strip():
                    st.error("Please add a title to your post.")
                elif not post_body.strip():
                    st.error("Please write some content for your post.")
                elif len(post_body.strip()) < 20:
                    st.error("Post content is too short. Write at least 20 characters.")
                else:
                    new_post = {
                        "id":        st.session_state.next_id,
                        "title":     post_title.strip(),
                        "body":      post_body.strip(),
                        "category":  post_cat,
                        "author":    user["name"],
                        "likes":     0,
                        "timestamp": str(time.time()),
                        "emoji":     post_emoji,
                        "user":      username,
                    }
                    st.session_state.posts_db.append(new_post)
                    st.session_state.next_id += 1
                    st.success("✅ Post published! Head to the Feed to see it.")
                    time.sleep(0.6)
                    st.session_state.page = "feed"
                    st.rerun()
        with can_col:
            if st.button("Cancel", use_container_width=True, key="cancel_btn"):
                st.session_state.page = "feed"
                st.rerun()


def show_profile():
    username = st.session_state.current_user
    user     = st.session_state.users_db[username]
    liked    = get_liked(username)
    disliked = get_disliked(username)
    acolor   = make_avatar_color(username)
    initials = get_initials(user["name"])
    my_posts = [p for p in st.session_state.posts_db if p.get("user") == username]
    total_likes_received = sum(p.get("likes",0) for p in my_posts)

    st.markdown("<div class='top-bar'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding-top:24px;margin-bottom:20px'>
        <div style='font-size:28px;font-weight:900;background:linear-gradient(135deg,#4f8ef7,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>My Profile 👤</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.2, 2])

    with left:
        st.markdown(f"""
        <div class='profile-card'>
            <div class='profile-avatar' style='background:linear-gradient(var(--surface),var(--surface)) padding-box,linear-gradient(135deg,{acolor},{acolor}88) border-box;color:{acolor}'>{initials}</div>
            <div class='profile-name'>{user["name"]}</div>
            <div class='profile-info'>@{username}</div>
            <div class='profile-info' style='margin-top:4px'>{user.get("bio","VIT Bhopal Student")}</div>
            <div class='profile-info' style='margin-top:4px'>🎓 {user.get("branch","CSE")} · {user.get("year","")}</div>
            <div class='profile-stats'>
                <div style='text-align:center'>
                    <div class='pstat-val'>{len(my_posts)}</div>
                    <div class='pstat-label'>Posts</div>
                </div>
                <div style='text-align:center'>
                    <div class='pstat-val'>{len(liked)}</div>
                    <div class='pstat-label'>Liked</div>
                </div>
                <div style='text-align:center'>
                    <div class='pstat-val'>{total_likes_received}</div>
                    <div class='pstat-label'>❤️ Got</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        vec, mat = build_tfidf(st.session_state.posts_db)
        if vec and mat is not None:
            agent = get_agent(username, mat.shape[1])
            if np.linalg.norm(agent.user_profile) > 0:
                st.markdown("""
                <div style='background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px;margin-top:0'>
                    <div style='font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>🧠 AI Interest Profile</div>""", unsafe_allow_html=True)
                fn   = vec.get_feature_names_out()
                idxs = agent.user_profile.argsort()[::-1][:10]
                kws  = [fn[i] for i in idxs if agent.user_profile[i] > 0]
                html = "<div class='int-cloud'>" + "".join(f"<span class='int-tag'>{k}</span>" for k in kws) + "</div></div>"
                st.markdown(html, unsafe_allow_html=True)

    with right:
        st.markdown(f"<div style='font-size:16px;font-weight:700;margin-bottom:14px'>📝 My Posts ({len(my_posts)})</div>", unsafe_allow_html=True)

        if not my_posts:
            st.markdown("""
            <div style='text-align:center;padding:40px;background:var(--surface);border:1px dashed var(--border);border-radius:16px;color:var(--muted)'>
                <div style='font-size:36px;margin-bottom:12px'>✏️</div>
                <div style='font-size:15px;color:var(--text)'>You haven't posted yet</div>
                <div style='font-size:13px;margin-top:6px'>Share something with the VIT community!</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Create Your First Post 🚀", use_container_width=True):
                st.session_state.page = "create"
                st.rerun()
        else:
            for p in sorted(my_posts, key=lambda x: x["id"], reverse=True):
                cfg = CAT_CONFIG.get(p["category"], CAT_CONFIG["General"])
                ts = p.get("timestamp","")
                try:
                    ts = time_ago(float(ts))
                except:
                    pass
                st.markdown(f"""
                <div class='post-card {cfg["card_class"]}'>
                    <div class='cbadge {cfg["badge_class"]}'>{cfg["icon"]} {p["category"]}</div>
                    <div class='post-title'>{p.get("emoji","💬")} {p["title"]}</div>
                    <div class='post-body'>{p["body"]}</div>
                    <div class='post-meta'>
                        <span>🕐 {ts}</span>
                        <span>❤️ {p.get("likes",0)} likes</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                dc, _ = st.columns([1,4])
                with dc:
                    if st.button("🗑️ Delete", key=f"pdel_{p['id']}"):
                        st.session_state.posts_db = [x for x in st.session_state.posts_db if x["id"] != p["id"]]
                        st.session_state.fv += 1
                        st.rerun()
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


st.markdown("<div class='top-bar'></div>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    show_auth()
else:
    show_sidebar()
    page = st.session_state.page
    if page == "feed":
        show_feed()
    elif page == "create":
        show_create_post()
    elif page == "profile":
        show_profile()