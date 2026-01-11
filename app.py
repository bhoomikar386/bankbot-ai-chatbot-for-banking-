import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from database.db import init_db, get_conn
from database.bank_crud import create_account, get_account, transfer_money
from database.security import verify_password
from dotenv import load_dotenv
from dialogue_manager.dialogue_handler import DialogueManager # Make sure this import is here

# Load environment variables (.env)
load_dotenv()

# Initialize Database
init_db()

#--- LLM INITIALIZATION ---
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=api_key
        )
    return None
# --- REALISTIC UI CONFIG ---
st.set_page_config(page_title="BHOOMIKA R", layout="wide", page_icon="💎")

def add_bg_and_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                        url("https://cms-resources.groww.in/uploads/TAX_SLAB_2025_06_03_T112143_053_11zon_980594947c.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }
        .premium-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }
        .stTextInput input, .stNumberInput input {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: black !important;
        }
        [data-testid="stDataFrame"] {
            background: white;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_and_style()

# --- SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- AUTHENTICATION ---
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<div class='premium-card' style='text-align:center;'>", unsafe_allow_html=True)
        st.title("🏦 BankBot Portal")
        t1, t2 = st.tabs(["🔐 Login", "📝 create new account"])
        
        with t1:
            acc = st.text_input("Account Number", key="login_acc")
            pwd = st.text_input("Password", type="password", key="login_pwd")
            if st.button("Access Account", use_container_width=True):
                user = get_account(acc)
                if user and verify_password(pwd, user[4]):
                    st.session_state.logged_in = True
                    st.session_state.user_name = user[1]
                    st.session_state.acc_no = acc
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        
        with t2:
            n_name = st.text_input("Full Name")
            n_acc = st.text_input("New Acc Number")
            n_bal = st.number_input("Initial Deposit", min_value=500)
            n_pwd = st.text_input("Set Password", type="password")
            if st.button("Create Account", use_container_width=True):
                create_account(n_name, n_acc, "Savings", n_bal, n_pwd)
                st.success("Registration Successful! Please log in.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # --- SIDEBAR ---
    with st.sidebar:
        name = st.session_state.user_name.lower()
       # Logic: If name contains any female keywords -> female.png, else default to male.png
        if any(x in name for x in ['kavya', 'kusuma', 'bhoomi']):
            avatar = "assets/female.png"
        else:
         avatar = "assets/male.png"
        st.image(avatar if os.path.exists(avatar) else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.write(f"### {st.session_state.user_name}")
        menu = st.radio("Menu", ["Dashboard", "AI Chatbot", "History", "Safety & Schemes", "Help"])
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # --- PAGES ---
    if menu == "Dashboard":
        u = get_account(st.session_state.acc_no)
        st.markdown(f"""
            <div class="premium-card">
                <h3>Welcome back, {st.session_state.user_name}</h3>
                <p style="font-size: 1.2em; opacity: 0.8;">Account Balance</p>
                <h1 style="color: #00d4ff; font-size: 3em;">${u[3]:,.2f}</h1>
                <hr>
                <p>Status: <span style="color:#28a745;">● Active</span> | Account No: {u[0]}</p>
            </div>
            """, unsafe_allow_html=True)

    elif menu == "AI Chatbot":
        st.title("🤖 Chat Assistant")



        
        # # Display Chat History
        # for msg in st.session_state.chat_history:
        #     with st.chat_message(msg["role"]):
        #         st.write(msg["content"])

        # prompt = st.chat_input("Check balance, Send money, or Block card...")
        # if prompt:
        #     st.session_state.chat_history.append({"role": "user", "content": prompt})
        #     st.rerun() # Refresh to show user message immediately

        # # Logical responses based on history
        # if st.session_state.chat_history:
        #     last_msg = st.session_state.chat_history[-1]
        #     if last_msg["role"] == "user":
        #         cmd = last_msg["content"].lower()

        
        # 1. Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 2. Get User Input
        prompt = st.chat_input("Check balance, Send money, or ask a question...")
        
        # 3. ONLY execute if prompt is not None and not empty
        if prompt:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Immediately show the user's message
            with st.chat_message("user"):
                st.write(prompt)

            cmd = prompt.lower()
            
            with st.chat_message("assistant"):
                # ROUTE A: BANKING ACTIONS
                if "balance" in cmd:
                    with st.form("bal_f"):
                        p = st.text_input("Verify Password", type="password")
                        if st.form_submit_button("Show Balance"):
                            u = get_account(st.session_state.acc_no)
                            if verify_password(p, u[4]):
                                st.success(f"✅ Your balance is: **${u[3]:,.2f}**")
                            else: st.error("Wrong Password")

                elif "send" in cmd or "transfer" in cmd:
                    with st.form("tx_f"):
                        to = st.text_input("Recipient Acc")
                        amt = st.number_input("Amount", min_value=1.0)
                        p = st.text_input("Pass", type="password")
                        if st.form_submit_button("Transfer"):
                            res = transfer_money(st.session_state.acc_no, to, amt, p)
                            st.info(res)


                # 4. GENERAL AI: GROQ LLM (For "What is Machine Learning", etc.)
                else:
                    llm = get_llm()
                    if llm:
                        with st.spinner("Consulting AI Engine..."):
                            response = llm.invoke([HumanMessage(content=prompt)])
                            st.write(response.content)
                            st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                    else:
                        st.error("LLM not configured. Check your API Key.")

                # 1. BANKING: CHECK BALANCE
            
                with st.chat_message("assistant"):
                    if "balance" in cmd:
                        with st.form("bal_f"):
                            p = st.text_input("Verify Password", type="password")
                            if st.form_submit_button("Show Balance"):
                                u = get_account(st.session_state.acc_no)
                                if verify_password(p, u[4]):
                                    st.balloons()
                                    st.success(f"✅ Your current balance is: **${u[3]:,.2f}**")
                                    st.toast("Balance fetched successfully!", icon="💰")
                                else:
                                    st.error("Incorrect Password.")

                    elif "send" in cmd or "transfer" in cmd:
                        with st.form("tx_f"):
                            to = st.text_input("Recipient Account Number")
                            amt = st.number_input("Amount ($)", min_value=1.0)
                            p = st.text_input("Enter Transaction Password", type="password")
                            if st.form_submit_button("Confirm Transfer"):
                                res = transfer_money(st.session_state.acc_no, to, amt, p)
                                if "Successful" in res:
                                    st.balloons()
                                    st.success(f"🎉 {res}")
                                    st.toast("Transaction Written to bankbot.db", icon="🚀")
                                else:
                                    st.error(res)

                    elif "block" in cmd:
                        with st.form("block_f"):
                            p = st.text_input("Enter Pass to BLOCK", type="password")
                            if st.form_submit_button("BLOCK CARD NOW"):
                                conn = get_conn()
                                conn.execute("UPDATE accounts SET account_type='BLOCKED' WHERE account_number=?", (st.session_state.acc_no,))
                                conn.commit()
                                st.snow()
                                st.error("🚫 ACCOUNT BLOCKED SUCCESSFULLY")
                                st.toast("Security Action Taken", icon="🚫")
                                st.balloons()
                             

                

    # --- HISTORY PAGE ---



    elif menu == "History":
        st.title("📜 Transactions")
        conn = get_conn()
        # Querying the actual bankbot.db file
        df = pd.read_sql_query("SELECT timestamp, from_account, to_account, amount FROM transactions WHERE from_account=? OR to_account=? ORDER BY timestamp DESC", 
                               conn, params=(st.session_state.acc_no, st.session_state.acc_no))
        conn.close()
        
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent transactions found in bankbot.db")

    elif menu == "Safety & Schemes":
        st.title("🛡️ Bank Safety")
        st.markdown("<div class='premium-card'><ul><li><b>Scheme A:</b> 7% Interest</li><li><b>Safety:</b> Never share OTP</li></ul></div>", unsafe_allow_html=True)
        st.video("https://www.youtube.com/shorts/9g-vzZUfZEQ")

    elif menu == "Help":
        st.title("❓ Help & Support")
        st.write("📞 Support: 1-800-BANK-BOT")
        st.write("📺 [YouTube Tutorial: Secure Banking](https://www.youtube.com/shorts/nC81-uPY7ak)")
        


        