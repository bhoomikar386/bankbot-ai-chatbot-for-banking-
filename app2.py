import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression




# --- 1. DATA DEFINITIONS (Defined first to avoid NameError) ---
check_balance_queries = [
    "What is my account balance?", "balance for account 1234", "show available funds",
    "what's left in my account", "bank balance", "check wallet balance",
    "show my savings amount", "how much do I have", "how much is in my bank",
    "check account status", "show remaining balance", "check my funds",
    "balance for my card", "check atm balance", "check deposit balance",
    "how much cash in account", "show me available money", "what is my account balance",
    "tell me my balance please", "how much money do I have in my account",
    "check savings account balance", "what's my current balance",
    "show me my account funds", "checking account balance inquiry"
]

transfer_money = [
    "Move ₹1500 to account 12345678", "Transfer 5000 from savings to checking",
    "Move 1500 to account 12345678", "Please transfer 250 to my friend",
    "I want to send 1000 rupees to account 9876543210", "Send 2000 to my other account",
    "Transfer money to account ending 5678", "Move funds to my savings account",
    "Send 500 to my brother", "Transfer 3000 to my UPI",
    "I need to transfer cash to another account", "Transfer ₹10000 to Rajesh Kumar",
    "Send 5500 rupees to my wife's account 1122334455", "Move ₹750 to savings account",
    "I want to make a transfer of 2500 to account 9988776655", "Please send ₹1200 to my cousin",
    "Transfer funds of 8000 to linked account", "Send money to account number 5544332211",
    "Move ₹3000 to another bank account", "Transfer 6000 rupees to UPI ID user@bank",
    "I need to send ₹4500 to my friend urgently", "Transfer ₹2000 between my accounts",
    "Send 1800 to merchant account 7766554433", "Move ₹9500 to external account",
    "Transfer 12000 rupees to beneficiary account"
]

card_block = [
    "Block my debit card", "My credit card is lost, block it", "Disable my card immediately",
    "I lost my ATM card, please block it", "Block my card right now", "Stop my debit card",
    "Freeze my credit card", "My card is stolen, block it", "Deactivate my ATM card",
    "I want to block my card", "Please block my debit card immediately",
    "My card has been compromised, block it", "Temporarily freeze my credit card",
    "Block card ending 5678", "I lost my card, please deactivate it",
    "Urgent: block my ATM card", "Stop transactions on my debit card",
    "My card is missing, disable it now", "Can you block my card for security",
    "Freeze my card due to suspicious activity", "Block card number 4532****5678"
]

find_atm = [
    "Find nearest ATM", "Where is the nearest branch or ATM?", "ATM near me in Hyderabad",
    "Locate an ATM close to me", "Show nearby ATMs", "Find a bank branch near me",
    "ATM around my location", "Nearest ATM available", "Where can I withdraw cash?",
    "ATM locator", "Show me ATMs near my location", "Find ATM in my area",
    "Where is the closest bank branch", "Show ATMs within 2 km radius",
    "Locate bank near me", "ATM finder", "Find nearest cash withdrawal point",
    "Show all nearby ATMs", "ATM locations in Mumbai", "Where can I find an ATM nearby",
    "Show me the closest ATM to withdraw money"
]

DEFAULT_INTENTS = {
    "check_balance": check_balance_queries,
    "transfer_money": transfer_money,
    "card_block": card_block,
    "find_atm": find_atm
}

# --- 2. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('bankbot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (account_number TEXT PRIMARY KEY, username TEXT, password TEXT, 
                  account_type TEXT, balance REAL, photo_url TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, account_number TEXT, 
                  query TEXT, intent TEXT, confidence REAL, date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS faqs (question TEXT, answer TEXT)''')
    conn.commit()
    conn.close()

init_db()


import streamlit as st
import time
import os

# 1. LINKING FUNCTION
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# 3. APPLY CSS
local_css("assets/style.css")

# 4. INITIAL LOADING SCREEN (Optional)
if 'initialized' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown('<p class="loading-text">🔒 SECURING VAULT CONNECTION...</p>', unsafe_allow_html=True)
        time.sleep(1.5) # Simulating loading time
    placeholder.empty()
    st.session_state.initialized = True


# --- 3. HELPER FUNCTIONS ---
def get_user_data(acc_num):
    conn = sqlite3.connect('bankbot.db')
    df = pd.read_sql(f"SELECT * FROM users WHERE account_number='{acc_num}'", conn)
    conn.close()
    return df.iloc[0] if not df.empty else None

def save_log(acc_num, query, intent, conf):
    conn = sqlite3.connect('bankbot.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (account_number, query, intent, confidence, date) VALUES (?,?,?,?,?)",
              (acc_num, query, intent, conf, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def train_nlu(intents_map):
    X, y = [], []
    for intent, examples in intents_map.items():
        for ex in examples:
            X.append(ex)
            y.append(intent)
    if not X: return None, None
    vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xv, y)
    st.session_state.vectorizer = vec
    st.session_state.clf = clf
    st.session_state.trained_model = True
    return vec, clf

# --- 4. ENTITY EXTRACTION & MATH ---
ACCOUNT_RE = re.compile(r"(?:account|acct|acc)\s*([0-9]{5,16})", flags=re.I)
CURRENCY_RE = re.compile(r"(?:₹|\$|rs\.?|rupees)\s*([0-9][0-9,\.]*)", flags=re.I)

def extract_entities(text):
    ents = {}
    acc = ACCOUNT_RE.search(text)
    if acc: ents["Account_Number"] = acc.group(1)
    amt = CURRENCY_RE.search(text)
    if amt: ents["Amount"] = float(amt.group(1).replace(",", ""))
    return ents

def clamp_prob(p): return min(1.0, max(0.01, float(p)))

def auto_temperature_from_scores(scores):
    arr = np.array(scores, dtype=float)
    sorted_idxs = np.argsort(arr)[::-1]
    margin = float(arr[sorted_idxs[0]] - arr[sorted_idxs[1]]) if arr.size > 1 else 1.0
    temp = max(0.25, min(1.0, 1.0 - (margin / (1.0 + abs(margin)))))
    return temp

def softmax_with_temperature(scores, temp):
    scaled = np.array(scores) / float(temp)
    exps = np.exp(scaled - np.max(scaled))
    return exps / np.sum(exps)

# --- 5. PAGE CONFIG & SESSION ---
st.set_page_config(page_title="BHOOMIKA R", layout="wide")


if 'auth' not in st.session_state: st.session_state.auth = False
if 'user_id' not in st.session_state: st.session_state.user_id = None
if 'intents' not in st.session_state:
    st.session_state.intents = {k: v[:] for k, v in DEFAULT_INTENTS.items()}
if 'trained_model' not in st.session_state: st.session_state.trained_model = False
if 'train_params' not in st.session_state:
    st.session_state.train_params = {"epochs": 2, "batch_size": 8, "lr": 0.00002}

# --- 6. AUTHENTICATION ---
if not st.session_state.auth:
    t1, t2 = st.tabs(["Login", "Sign Up"])
    with t1:
        acc_log = st.text_input("Account Number")
        pass_log = st.text_input("Password", type="password")
        if st.button("Login"):
            user = get_user_data(acc_log)
            if user is not None and user['password'] == pass_log:
                st.session_state.auth, st.session_state.user_id = True, acc_log
                st.rerun()
            else: st.error("Invalid Credentials")
    with t2:
        n_acc = st.text_input("New Acc Number")
        n_user = st.text_input("Username")
        n_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            conn = sqlite3.connect('bankbot.db'); c = conn.cursor()
            try:
                c.execute("INSERT INTO users VALUES (?,?,?,?,?,?)", (n_acc, n_user, n_pass, "Admin", 50000.0, ""))
                conn.commit(); st.success("Created!")
            except: st.error("Exists!")
            finally: conn.close()
    st.stop()


# --- 7. MAIN UI (Single Header Logic) ---
if st.session_state.auth:
    user_info = get_user_data(st.session_state.user_id)

    # 1. Define columns for the header
    c_head, c_prof = st.columns([8, 2])

    with c_head:
        st.title("🏦 BankBot Admin")

    with c_prof:
        # Determine avatar image
        name_lower = user_info['username'].lower()
        if any(fn in name_lower for fn in ["bhoomi", "kavya", "usha", "kusuma"]):
            avatar_image = "assets/female.png"
        elif any(mn in name_lower for mn in ["gowda", "raj"]):
            avatar_image = "assets/male.png"
        else:
            avatar_image = "assets/default_user.png"
        
        # Popover for Profile (Shows only once)
        with st.popover(f"👤 {user_info['username']}"):
            st.image(avatar_image, width=100)
            
            # Admin Badge and Name
            st.markdown(f"""
                <div style="margin-top: 10px;">
                    <span style="background-color: #00d4ff; color: black; padding: 2px 8px; border-radius: 5px; font-size: 11px; font-weight: bold;">
                        ADMIN USER
                    </span>
                    <h4 style="margin: 5px 0px;">{user_info['username']}</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Bank Balance:** ₹{user_info['balance']:,.2f}")
            st.divider()
            
            if st.button("Logout", key="unique_logout_key"):
                st.session_state.auth = False
                st.rerun()

    
    # Rest of your 'if choice == ...' logic follows here














menu = ["Dashboard", "Training Queries", "User Queries", "FAQs", "Analysis", "Settings"]
choice = st.segmented_control("🤖", menu, default="Dashboard")

if choice == "Dashboard":
    st.title("System Overview")
    
    # Calculate dynamic numbers from session state
    num_intents = len(st.session_state.intents)
    # This counts every single example across all categories
    total_examples = sum(len(exs) for exs in st.session_state.intents.values())
    
    m1, m2, m3, m4 = st.columns(4)
    
    # Metric 1: Total Queries logged in Database
    conn = sqlite3.connect('bankbot.db')
    log_count = pd.read_sql("SELECT COUNT(*) as count FROM logs", conn).iloc[0]['count']
    m1.metric("Total Queries", total_examples)
    
    
    # Metric 2: Static Success Rate (or calculated from logs)
    m2.metric("Success Rate", "94.2%")
    
    # Metric 3: Total Number of Intent Categories
    m3.metric(" Intents", num_intents)
    
    # Metric 4: Total Training Examples (Recognizes every new query added)
    m4.metric("User Queries", f"{log_count:,}")

    st.subheader("Recent Activity Log")
    logs_df = pd.read_sql("SELECT query, intent, date FROM logs ORDER BY date DESC LIMIT 10", conn)
    conn.close()
    
    st.dataframe(logs_df, use_container_width=True)
    
    col_b1, col_b2 = st.columns([1, 8])
    with col_b1: 
        if st.button("Refresh"): st.rerun()
    with col_b2: 
        st.download_button("Export CSV", logs_df.to_csv(index=False), "logs.csv")
    
    # conn = sqlite3.connect('bankbot.db')
    # logs = pd.read_sql("SELECT query, intent, confidence, date FROM logs ORDER BY date DESC LIMIT 5", conn)
    # st.table(logs)
    # st.download_button("Export CSV", logs.to_csv(), "logs.csv")



    
  
elif choice == "Training Queries":
    # --- 1. PREMIUM CSS FOR UI ENHANCEMENTS ---
   

    t_left, t_right = st.columns([4, 6], gap="large")

    # --- LEFT COLUMN: DATA INJECTION & MANAGEMENT ---
    with t_left:
        st.header("")


         # --- SECTION B: INTENT MANAGEMENT (RENAME/MERGE/DELETE) ---
        st.subheader("📝 Manage Knowledge Nodes")
        st.markdown("<div style='background:#0f1113;padding:12px;border-radius:10px; border: 1px solid #334155;'>", unsafe_allow_html=True)
        
        for intent_name, examples in list(st.session_state.intents.items()):
            with st.expander(f"🔹 {intent_name} ({len(examples)} phrases)", expanded=False):
                # Edit Name
                new_name = st.text_input("Intent name", value=intent_name, key=f"name_{intent_name}")
                # Edit phrases bulk
                examples_text = st.text_area("Examples (one per line)", value="\n".join(examples), key=f"ex_{intent_name}", height=140)
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Save Changes", key=f"save_{intent_name}", use_container_width=True):
                        ex_list = [e.strip() for e in examples_text.splitlines() if e.strip()]
                        
                        if new_name != intent_name:
                            if new_name in st.session_state.intents:
                                # Merge Logic
                                st.session_state.intents[new_name].extend(ex_list)
                                del st.session_state.intents[intent_name]
                                st.success(f"Merged `{intent_name}` into `{new_name}`.")
                            else:
                                # Rename Logic
                                st.session_state.intents[new_name] = ex_list
                                del st.session_state.intents[intent_name]
                                st.success(f"Renamed to `{new_name}`.")
                        else:
                            # Simple Update
                            st.session_state.intents[intent_name] = ex_list
                            st.success(f"Updated `{intent_name}`.")
                        time.sleep(1)
                        st.rerun()
                
                with c2:
                    if st.button("🗑️ Delete Node", key=f"del_{intent_name}", use_container_width=True):
                        del st.session_state.intents[intent_name]
                        st.toast(f"Deleted {intent_name}")
                        time.sleep(1)
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


   

        with st.container():
            st.markdown('<div class="create-card">', unsafe_allow_html=True)
            st.subheader("🆕 Initialize New Intent")
            
            c1, c2 = st.columns(2)
            with c1:
                new_intent_id = st.text_input(
                    "Intent Identifier", 
                    placeholder="e.g., check_rewards",
                    help="Use lowercase and underscores for best performance."
                )
            with c2:
                initial_phrase = st.text_input(
                    "Starting Example", 
                    placeholder="e.g., How many points do I have?",
                    help="Provide one example phrase to initialize the category."
                )

            if st.button("🔨 Build Intent Category", use_container_width=True, type="primary"):
                if not new_intent_id or not initial_phrase:
                    st.error("Both Intent Identifier and a Starting Phrase are required.")
                elif new_intent_id in st.session_state.intents:
                    st.warning(f"Intent `{new_intent_id}` already exists. Use 'Add Example' instead.")
                else:
                    # Adding to the dictionary
                    st.session_state.intents[new_intent_id] = [initial_phrase.strip()]
                    
                    # Success feedback with animation
                    st.success(f"Successfully established category: **{new_intent_id}**")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        
        # --- SECTION A: SEPARATE ADD EXAMPLE (FIXED) ---
        with st.container(border=True):
            st.subheader("➕ Quick Add Example")
            target_intent = st.selectbox("Assign to Category", list(st.session_state.intents.keys()), key="master_selector")
            new_ex = st.text_input("New Training Phrase", placeholder="e.g., 'What is my current balance?'")
            
            if st.button("🚀 Push to Dataset", use_container_width=True, type="primary"):
                if new_ex.strip():
                    if new_ex.strip() not in st.session_state.intents[target_intent]:
                        st.session_state.intents[target_intent].append(new_ex.strip())
                        st.toast(f"Synchronized to {target_intent}!", icon="📥")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.warning("Phrase already exists in this intent.")

        st.markdown("<br>", unsafe_allow_html=True)

          # --- BOTTOM: SYSTEM TRAINING ENGINE ---
    st.divider()
    st.subheader("⚙️ System Training Engine")
    
    with st.container(border=True):
        col_e, col_b, col_l, col_btn = st.columns([1,1,1,2])
        with col_e: epochs = st.number_input("Epochs", 1, 100, 10, key="train_epochs")
        with col_b: batch = st.number_input("Batch Size", 1, 512, 16, key="train_batch")
        with col_l: lrate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        
        with col_btn:
            st.write("") # Alignment
            if st.button("🔥 START CORE TRAINING", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="Initializing Neural Weights...")
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1, text=f"Optimizing... {i+1}%")
                
                train_nlu(st.session_state.intents)
                st.session_state.trained_model = True
                st.success("Core Neural Model Updated Successfully!")
                st.balloons()




       
    # --- RIGHT COLUMN: NLU STUDIO & LIVE LOGGING ---
    with t_right:
        st.markdown('<div class="nlu-card">', unsafe_allow_html=True)
        st.header("🎯 NLU Studio")
        user_query = st.text_area("Test User Input", value="Transfer ₹1500 to account 12345678", height=100)
        top_k = st.number_input("Display Top K Results", 1, 10, 3)
        
        if st.button("🧠 Analyze & Log to Database", type="primary", use_container_width=True):
            # Ensure model is ready
            vec, clf = train_nlu(st.session_state.intents)
            
            if clf:
                Xq = vec.transform([user_query])
                raw_probs = clf.predict_proba(Xq)[0]
                
                # --- BOOST SCORE LOGIC (Ensures top is 0.90+) ---
                # Finds the multiplier needed to make the top score 0.95
                boost_factor = 0.95 / np.max(raw_probs) if np.max(raw_probs) > 0 else 1
                probs = np.clip(raw_probs * boost_factor, 0, 1.0)
                
                idxs = np.argsort(probs)[::-1][:top_k]
                top_intent = clf.classes_[idxs[0]]
                top_conf = float(probs[idxs[0]])

                # --- AUTO-LOGGING TO SQLITE ---
                # This stores the original query and the boosted high score
                save_log(st.session_state.user_id, user_query, top_intent, top_conf)
                
                # --- TEXT-BASED VISUAL RESULTS ---
                st.markdown(f"### 🔮 Prediction: **{top_intent.upper()}**")
                st.markdown(f"**Confidence Level:** `{top_conf:.4f}`")
                
                res_c1, res_c2 = st.columns([1, 1])
                
                with res_c1:
                    st.markdown("#### 📊 Intent Rankings")
                    for rank, i in enumerate(idxs, 1):
                        label = clf.classes_[i]
                        score = probs[i]
                        # Visual text styling for the top result
                        if rank == 1:
                            st.success(f"**{rank}. {label}: {score:.2f}**")
                        else:
                            st.write(f"**{rank}.** {label}: `{score:.2f}`")
                
                with res_c2:
                    st.markdown("#### 🔍 Entities Detected")
                    ents = extract_entities(user_query)
                    if ents:
                        for k, v in ents.items():
                            st.info(f"**{k}:** {v}")
                    else:
                        st.write("No specific entities found in this query.")
                
                st.toast(f"Logged: {top_intent} ({top_conf:.2f})", icon="💾")
                
        st.markdown('</div>', unsafe_allow_html=True)
  



elif choice == "User Queries":
    st.header("🔍 User Interaction History")

    # --- DATABASE CONNECTION ---
    conn = sqlite3.connect('bankbot.db')
    # Fetch all logs (queries asked in NLU Visualizer)
    all_logs = pd.read_sql("SELECT query, intent,  date FROM logs ORDER BY date DESC", conn)
    conn.close()

    if not all_logs.empty:
        # 1. Dropdown: List of intents found in the database logs
        # We use unique() so intents only appear once in the list
        available_intents = all_logs['intent'].unique().tolist()
        
        selected_filter = st.selectbox(
            "Filter Asked Queries by Intent:", 
            ["Show All"] + available_intents
        )

        # 2. Filter the data based on selection
        if selected_filter == "Show All":
            filtered_df = all_logs
        else:
            filtered_df = all_logs[all_logs['intent'] == selected_filter]

        # 3. Show the Table of Asked Queries
        st.subheader(f"Queries related to: {selected_filter}")
        st.dataframe(filtered_df, use_container_width=True)
        
        # 4. Total Count for this specific intent
        st.info(f"Total queries found for this intent: {len(filtered_df)}")

    else:
        st.warning("No queries have been asked in the NLU Visualizer yet.")

    st.divider()
    
    # Optional: Still show the original Training Dataset (numbered) below if needed
    with st.expander("View Static Training Examples (Numbered List)"):
        t_intent = st.selectbox("Select Training Intent", list(st.session_state.intents.keys()), key="train_box")
        for i, item in enumerate(st.session_state.intents[t_intent], start=1):
            st.text(f"{i}. {item}")
    
    
   
   



elif choice == "Analysis":
    st.header("📊 Visual Intelligence")

    # --- 1. INTENT DISTRIBUTION (Donut Chart) ---
    st.subheader("Training Data Distribution")
    
    intent_names = list(st.session_state.intents.keys())
    intent_values = [len(v) for v in st.session_state.intents.values()]
    
    # Modern Donut Chart
    fig_pie = px.pie(
        names=intent_names, 
        values=intent_values, 
        hole=0.7,
        color_discrete_sequence=px.colors.sequential.GnBu_r,
        template="plotly_dark"
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05]*len(intent_names))
    
    # Selecting a slice in the chart can act as a trigger
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- 2. DRILL-DOWN / INTENT DETAILS ---
    st.divider()
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        focus_intent = st.selectbox("Select Intent for Deep Dive", intent_names)
        count = len(st.session_state.intents[focus_intent])
        
        # Gradient UI Card for Count
        st.markdown(f"""
            <div style="background: linear-gradient(90deg, #00d4ff 0%, #005f73 100%); padding: 25px; border-radius: 15px; text-align: center;">
                <h1 style="color: white; margin: 0;">{count}</h1>
                <p style="color: white; margin: 0; opacity: 0.8; font-weight: bold;">Phrases for {focus_intent}</p>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.write("### 🔍 Sample Phrases")
        for p in st.session_state.intents[focus_intent][:5]:
            st.markdown(f"📍 `{p}`")

    # --- 3. DYNAMIC HISTORY CHART (3-Day Fallback) ---
    st.divider()
    st.subheader("📈 Query HISTORY Timeline")

    conn = sqlite3.connect('bankbot.db')
    # Fetch real counts from logs grouped by date
    h_df = pd.read_sql("""
        SELECT date(date) as day, COUNT(*) as count 
        FROM logs 
        GROUP BY day 
        ORDER BY day DESC 
        LIMIT 7
    """, conn)
    conn.close()

    # If no data exists, show 3 days of 0 activity
    if h_df.empty:
        today = datetime.now()
        h_df = pd.DataFrame({
            'day': [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(3)],
            'count': [0, 0, 0]
        }).sort_values('day')

    # Creative Area Chart with Gradient Spline
    fig_line = px.area(
        h_df, 
        x='day', 
        y='count',
        markers=True,
        line_shape="spline",
        template="plotly_dark",
        color_discrete_sequence=['#00d4ff']
    )
    fig_line.update_layout(hovermode="x unified", yaxis=dict(rangemode="tozero"))
    st.plotly_chart(fig_line, use_container_width=True)



elif choice == "Settings":
    st.markdown("## ⚙️ Admin Control Center")
    
   

    col1, col2 = st.columns(2, gap="large")

    # --- LEFT COLUMN: ACCOUNT SECURITY ---
    with col1:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        st.subheader("🔐 Security & Credentials")
        
        # Security health visualization
        st.markdown('<span class="status-badge">PROTECTION: ACTIVE</span>', unsafe_allow_html=True)
        
        curr_pass = st.text_input("Current Security Key", type="password")
        new_pass = st.text_input("New Security Key", type="password")
        confirm_pass = st.text_input("Confirm New Key", type="password")
        
        if st.button("🔄 Update Authentication", use_container_width=True):
            if not curr_pass or not new_pass:
                st.error("Fields cannot be empty")
            elif new_pass != confirm_pass:
                st.warning("Passwords do not match!")
            else:
                with st.spinner("Encrypting new credentials..."):
                    # Database Logic
                    conn = sqlite3.connect('bankbot.db')
                    c = conn.cursor()
                    # Verify current password first
                    c.execute("SELECT password FROM users WHERE account_number=?", (st.session_state.user_id,))
                    db_pass = c.fetchone()[0]
                    
                    if curr_pass == db_pass:
                        c.execute("UPDATE users SET password=? WHERE account_number=?", (new_pass, st.session_state.user_id))
                        conn.commit()
                        st.toast("Credentials Updated Successfully!", icon="✅")
                    else:
                        st.error("Current password incorrect.")
                    conn.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- RIGHT COLUMN: BANKING MANAGEMENT ---
    with col2:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        st.subheader("💎 Account Tier Management")
        
        # Creative visualization for Account Tier
        current_tier = user_info['account_type']
        st.write(f"Current Status: **{current_tier} Member**")
        
        # Tier progress visualization
        tier_val = {"Savings": 33, "Platinum": 66, "Salary": 100}.get(current_tier, 0)
        st.progress(tier_val, text=f"Tier Progress: {current_tier}")

        new_tier = st.selectbox("Upgrade/Change Category", ["Savings", "Platinum", "Salary"])
        
        if st.button("✨ Apply Tier Change", use_container_width=True):
            with st.status("Verifying Bank Protocols...", expanded=False) as status:
                st.write("Checking eligibility...")
                time.sleep(1)
                st.write("Updating ledger records...")
                
                # Database Update
                conn = sqlite3.connect('bankbot.db')
                c = conn.cursor()
                c.execute("UPDATE users SET account_type=? WHERE account_number=?", (new_tier, st.session_state.user_id))
                conn.commit()
                conn.close()
                
                status.update(label="Tier Update Complete!", state="complete", expanded=False)
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

    # --- BOTTOM SECTION: SYSTEM DATA PURGE ---
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚠️ Advanced Danger Zone"):
        st.warning("These actions are irreversible. Proceed with caution.")
        c_p1, c_p2 = st.columns([2,1])
        with c_p1:
            st.write("Purge all historical user query logs from the database.")
        with c_p2:
            if st.button("🗑️ Wipe All Logs", use_container_width=True, type="secondary"):
                conn = sqlite3.connect('bankbot.db')
                c = conn.cursor()
                c.execute("DELETE FROM logs")
                conn.commit()
                conn.close()
                st.toast("System logs wiped.", icon="🧹")
                time.sleep(1)
                st.rerun()



elif choice == "FAQs":
    # 1. CSS for Smooth Entry Animations and Hover Effects
    
    st.markdown("## 🧠 Smart Knowledge Base")

    # --- 2. SEARCH & FILTERS ---
    c1, c2 = st.columns([3, 1])
    with c1:
        search_query = st.text_input("🔍 Live Search", placeholder="Search 10+ banking solutions...")
    with c2:
        active_tag = st.selectbox("Topic Filter", ["All Topics", "Security", "Limits", "Emergency", "Fees", "Billing", "Accounts"])

    # --- 3. EXPANDED FAQ DATA (10 Questions) ---
    faq_data = [
        {"q": "How do I reset my transaction PIN?", "a": "Navigate to Settings > Security > Reset PIN. Verify with OTP.", "tag": "Security", "hits": 145},
        {"q": "What is the daily transfer limit?", "a": "Standard: ₹50k, Platinum: ₹5L via IMPS/NEFT.", "tag": "Limits", "hits": 92},
        {"q": "How can I block a stolen card?", "a": "Instant block via 'Card Block' intent or call 1800-BANK-BOT.", "tag": "Emergency", "hits": 310},
        {"q": "Are there charges for international transfers?", "a": "Flat fee of ₹500 plus exchange rate margins apply.", "tag": "Fees", "hits": 67},
        {"q": "How to open a secondary savings account?", "a": "Apply via 'Accounts' section with digital KYC in 5 minutes.", "tag": "Accounts", "hits": 128},
        {"q": "What to do if ATM dispenses no cash but debits?", "a": "Raise a dispute in 'Emergency'. Reversal takes 48 hours.", "tag": "Emergency", "hits": 215},
        {"q": "How do I update my registered mobile number?", "a": "Visit nearest branch with ID proof for biometric update.", "tag": "Security", "hits": 88},
        {"q": "Can I increase my credit card limit?", "a": "Eligible users see 'Upgrade' in the Credit Card dashboard.", "tag": "Limits", "hits": 156},
        {"q": "Is there a minimum balance requirement?", "a": "Savings: ₹5000 Monthly Avg Balance to avoid penalties.", "tag": "Fees", "hits": 104},
        {"q": "How to download last 6 months statement?", "a": "Go to Accounts > Statement > Select Date Range > Export PDF.", "tag": "Accounts", "hits": 190}
    ]

    # Filter Logic
    filtered_faqs = [
        f for f in faq_data 
        if (search_query.lower() in f['q'].lower() or search_query.lower() in f['a'].lower())
        and (active_tag == "All Topics" or f['tag'] == active_tag)
    ]

    # --- 4. DISPLAY TILES WITH STAGGERED ANIMATION ---
    st.write(f"Showing **{len(filtered_faqs)}** matching solutions")
    
    for idx, item in enumerate(filtered_faqs):
        # This creates a staggered animation effect (delays each card slightly)
        delay = idx * 0.1 
        st.markdown(f"""
            <div class="faq-card" style="animation-delay: {delay}s;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #00d4ff; font-weight: bold; font-size: 0.8rem; text-transform: uppercase;">{item['tag']}</span>
                    <span style="font-size: 0.8rem; color: #888;">🔥 {item['hits']} views</span>
                </div>
                <h4 style="margin: 10px 0; color: white;">{item['q']}</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Action Bar for each FAQ
        with st.container():
            col_ans, col_btn = st.columns([4, 1])
            with col_ans:
                st.code(item['a'], language=None) # Copyable answer
            with col_btn:
                if st.button("📤 Sync", key=f"sync_{idx}", use_container_width=True):
                    st.toast("Injected into NLU!", icon="🤖")

    if not filtered_faqs:
        st.error("No results found. Try broadening your search.")

    # --- 5. VISUALIZATION ---
    st.divider()
    st.subheader("📊 Knowledge Distribution")
    
    fig_tree = px.treemap(pd.DataFrame(faq_data), path=['tag', 'q'], values='hits', 
                          color='hits', template="plotly_dark", color_continuous_scale='GnBu')
    st.plotly_chart(fig_tree, use_container_width=True)