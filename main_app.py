import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import json
import numpy as np
import time

from typer import style

st.set_page_config(page_title="BHOOMIKA R", layout="wide")

# CSS
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --bg-main: #0b0f12;
        --bg-card: #11151a;
        --bg-soft: #161b22;
        --accent: #00e5ff;
        --accent-soft: #1de9b6;
        --text-main: #e6eef3;
        --text-muted: #9aa4ad;
    }
   

    /* Main title */
    .big-title, h1, h2, h3 {
        background: linear-gradient(90deg, var(--accent), var(--accent-soft));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: 0.4px;
    }

    /* Cards */
    .right-card, 
    div[data-testid="stExpander"],
    div[data-testid="stVerticalBlock"] > div {
        background: linear-gradient(180deg, #11161d, #0d1218);
        border-radius: 14px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.45);
        transition: all 0.25s ease-in-out;
    }

    /* Hover effect */
    .right-card:hover,
    div[data-testid="stExpander"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 40px rgba(0,229,255,0.15);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        color: #ffffff !important;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px;
        margin-bottom: 6px;
    }

    /* Text inputs */
    input, textarea {
        background: var(--bg-soft) !important;
        color: var(--text-main) !important;
        border-radius: 10px !important;
        border: 1px solid #1f2a36 !important;
        padding: 10px !important;
    }

    /* Buttons */
    button[kind="primary"], 
    .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-soft)) !important;
        color: #002b36 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 10px 18px !important;
        transition: all 0.25s ease-in-out;
    }

    button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0,229,255,0.6);
    }

    /* Badges */
    .badge-true {
        background: linear-gradient(90deg, #00c853, #b2ff59);
        color: #002b1f;
        font-weight: 700;
        border-radius: 20px;
        padding: 6px 14px;
    }

    /* Success / info / warning tweaks */
    .stAlert {
        border-radius: 12px;
        border-left: 6px solid var(--accent);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--accent), var(--accent-soft));
        border-radius: 6px;
    }

    /* Section spacing */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1016, #05080c);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Default intents
check_balance_queries = [
    "What is my account balance?",
    "balance for account 1234",
    "show available funds",
    "what's left in my account",
    "bank balance",
    "check wallet balance",
    "show my savings amount",
    "how much do I have",
    "how much is in my bank",
    "check account status",
    "show remaining balance",
    "check my funds",
    "balance for my card",
    "check atm balance",
    "check deposit balance",
    "how much cash in account",
    "show me available money",
    "what is my account balance",
    "tell me my balance please",
    "how much money do I have in my account",
    "check savings account balance",
    "what's my current balance",
    "show me my account funds",
    "checking account balance inquiry"
]

transfer_money = [
    "Move ₹1500 to account 12345678",
    "Transfer 5000 from savings to checking",
    "Move 1500 to account 12345678",
    "Please transfer 250 to my friend",
    "I want to send 1000 rupees to account 9876543210",
    "Send 2000 to my other account",
    "Transfer money to account ending 5678",
    "Move funds to my savings account",
    "Send 500 to my brother",
    "Transfer 3000 to my UPI",
    "I need to transfer cash to another account",
    "Transfer ₹10000 to Rajesh Kumar",
    "Send 5500 rupees to my wife's account 1122334455",
    "Move ₹750 to savings account",
    "I want to make a transfer of 2500 to account 9988776655",
    "Please send ₹1200 to my cousin",
    "Transfer funds of 8000 to linked account",
    "Send money to account number 5544332211",
    "Move ₹3000 to another bank account",
    "Transfer 6000 rupees to UPI ID user@bank",
    "I need to send ₹4500 to my friend urgently",
    "Transfer ₹2000 between my accounts",
    "Send 1800 to merchant account 7766554433",
    "Move ₹9500 to external account",
    "Transfer 12000 rupees to beneficiary account"
]

card_block = [
    "Block my debit card",
    "My credit card is lost, block it",
    "Disable my card immediately",
    "I lost my ATM card, please block it",
    "Block my card right now",
    "Stop my debit card",
    "Freeze my credit card",
    "My card is stolen, block it",
    "Deactivate my ATM card",
    "I want to block my card",
    "Please block my debit card immediately",
    "My card has been compromised, block it",
    "Temporarily freeze my credit card",
    "Block card ending 5678",
    "I lost my card, please deactivate it",
    "Urgent: block my ATM card",
    "Stop transactions on my debit card",
    "My card is missing, disable it now",
    "Can you block my card for security",
    "Freeze my card due to suspicious activity",
    "Block card number 4532****5678"
]

find_atm = [
    "Find nearest ATM",
    "Where is the nearest branch or ATM?",
    "ATM near me in Hyderabad",
    "Locate an ATM close to me",
    "Show nearby ATMs",
    "Find a bank branch near me",
    "ATM around my location",
    "Nearest ATM available",
    "Where can I withdraw cash?",
    "ATM locator",
    "Show me ATMs near my location",
    "Find ATM in my area",
    "Where is the closest bank branch",
    "Show ATMs within 2 km radius",
    "Locate bank near me",
    "ATM finder",
    "Find nearest cash withdrawal point",
    "Show all nearby ATMs",
    "ATM locations in Mumbai",
    "Where can I find an ATM nearby",
    "Show me the closest ATM to withdraw money"
]

DEFAULT_INTENTS = {
    "check_balance": check_balance_queries,
    "transfer_money": transfer_money,
    "card_block": card_block,
    "find_atm": find_atm
}

# Session state
if "intents" not in st.session_state:
    st.session_state.intents = {k: v[:] for k, v in DEFAULT_INTENTS.items()}
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "clf" not in st.session_state:
    st.session_state.clf = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = False
if "train_params" not in st.session_state:
    st.session_state.train_params = {"epochs": 2, "batch_size": 8, "lr": 0.00002}

# Train function (slightly stronger regularization options)
def train_nlu(intents_map):
    X = []
    y = []
    for intent, examples in intents_map.items():
        for ex in examples:
            X.append(ex)
            y.append(intent)
    if not X:
        return None, None
    # use sublinear_tf and ngram up to 2 to improve signal
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000, sublinear_tf=True)
    Xv = vec.fit_transform(X)
    # slightly stronger regularization allowing model to be more confident (tune C if needed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xv, y)
    st.session_state.vectorizer = vec
    st.session_state.clf = clf
    st.session_state.trained_model = True
    return vec, clf

# ----------------- FIXED Entity extraction -----------------
ACCOUNT_RE = re.compile(r"(?:account|acct|a/c|a c|a\.c\.|acc)\s*(?:no[:#]?\s*)?[:#]?\s*([0-9]{5,16})", flags=re.I)
FALLBACK_DIGIT_RE = re.compile(r"\b([0-9]{6,16})\b")

CURRENCY_NUM_RE = re.compile(
    r"(?:₹|\$|rs\.?|rupees|rupee|inr)\s*([0-9][0-9,\.]*)", flags=re.I
)
NUM_CURRENCY_WORD_RE = re.compile(
    r"([0-9][0-9,\.]*)\s*(?:rupees|rupee|rs\.?|inr|\$)", flags=re.I
)

def _clean_number_str(s: str) -> str:
    return re.sub(r"[,\s]", "", s)

def extract_entities(text: str):
    ents = {}
    # Account extraction
    acc = ACCOUNT_RE.search(text)
    if acc:
        ents["Account_Number"] = acc.group(1)
    else:
        fb = FALLBACK_DIGIT_RE.findall(text)
        if fb:
            ents["Account_Number"] = sorted(fb, key=lambda x: -len(x))[0]

    t = text
    m = CURRENCY_NUM_RE.search(t)
    if m:
        num_str = m.group(1)
        cleaned = _clean_number_str(num_str)
        try:
            ents["Amount"] = float(cleaned)
            return ents
        except:
            pass
    m2 = NUM_CURRENCY_WORD_RE.search(t)
    if m2:
        num_str = m2.group(1)
        cleaned = _clean_number_str(num_str)
        try:
            ents["Amount"] = float(cleaned)
            return ents
        except:
            pass
    return ents

# Helper
def clamp_prob(p):
    p_clamped = max(0.01, float(p))
    return min(1.0, p_clamped)

# Softmax w/ temperature
def softmax_with_temperature(scores, temperature):
    scaled = np.array(scores) / float(max(1e-6, temperature))
    exps = np.exp(scaled - np.max(scaled))
    probs = exps / np.sum(exps)
    return probs

# Automatic temperature selector based on margin between top-2
def auto_temperature_from_scores(scores):
    """
    Given raw scores (logits) or prob estimates, compute a temperature in [0.25, 1.0]
    Lower temperature -> sharper distribution. We map margin -> temperature.
    """
    arr = np.array(scores, dtype=float)
    # normalize to 0..1 by softmax then get margin
    # but if scores are logits, margin = top1 - top2 is already meaningful
    sorted_idxs = np.argsort(arr)[::-1]
    top1 = arr[sorted_idxs[0]]
    top2 = arr[sorted_idxs[1]] if arr.size > 1 else -np.inf
    margin = float(top1 - top2)
    # clamp margin to [0, +inf). Use a mapping that yields smaller temps for larger margins.
    # margin typically small; map via: temp = max(0.25, min(1.0, 1.0 - 0.6 * sigmoid(margin_scaled)))
    # We'll scale margin with softplus-like transform for stability:
    margin_scaled = margin / (1.0 + abs(margin))  # maps to (-1,1)
    # Now map to temperature in [0.25,1.0]. Larger margin -> smaller temp.
    temp = 1.0 - 0.75 * margin_scaled  # if margin_scaled near 1 -> temp ~ 0.25
    temp = max(0.25, min(1.0, temp))
    return temp

# When clf doesn't have decision_function (rare for LR), fallback sharpen via exponent
def sharpen_probs_by_margin(raw_probs, margin):
    """
    raw_probs: np.array of class probabilities
    margin: top1 - top2
    returns sharpened probs
    """
    # map margin to exponent factor gamma >= 1.0
    gamma = 1.0 + max(0.0, margin) * 3.0  # margin 0 -> gamma=1 ; margin 0.5 -> gamma=2.5
    powered = np.power(np.maximum(raw_probs, 1e-12), gamma)
    return powered / np.sum(powered)

# UI
st.markdown('<div style="font-size:28px;font-weight:700;margin-bottom:6px;">BankBot NLU – Intent & Entity Engine</div>', unsafe_allow_html=True)
left_col, right_col = st.columns([2,3])

with left_col:
    st.markdown("<div style='background:#0f1113;padding:12px;border-radius:6px;'>", unsafe_allow_html=True)
    for intent_name, examples in list(st.session_state.intents.items()):
        with st.expander(f"{intent_name} ({len(examples)} examples)", expanded=False):
            new_name = st.text_input("Intent name", value=intent_name, key=f"name_{intent_name}")
            examples_text = st.text_area("Examples (one per line)", value="\n".join(examples), key=f"ex_{intent_name}", height=140)
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("Save changes", key=f"save_{intent_name}"):
                    ex_list = [e.strip() for e in examples_text.splitlines() if e.strip()]
                    if new_name != intent_name:
                        if new_name in st.session_state.intents:
                            st.session_state.intents[new_name].extend(ex_list)
                            del st.session_state.intents[intent_name]
                            st.success(f"Merged `{intent_name}` into existing `{new_name}`.")
                        else:
                            st.session_state.intents[new_name] = ex_list
                            del st.session_state.intents[intent_name]
                            st.success(f"Renamed `{intent_name}` → `{new_name}`.")
                    else:
                        st.session_state.intents[intent_name] = ex_list
                        st.success(f"Updated `{intent_name}` examples ({len(ex_list)}).")
            with col2:
                if st.button("Delete intent", key=f"del_{intent_name}"):
                    del st.session_state.intents[intent_name]
                    st.experimental_rerun()
    st.markdown("---")
    st.subheader("Add new intent")
    with st.form("add_intent_form", clear_on_submit=True):
        ai_name = st.text_input("Intent name (snake_case)", placeholder="transfer_money")
        ai_examples = st.text_area("Examples (one per line)", placeholder="Transfer ₹500 to account 12345678\nMove 1500 rupees")
        submitted = st.form_submit_button("Add Intent")
        if submitted:
            if not ai_name.strip():
                st.error("Provide an intent name.")
            else:
                exs = [e.strip() for e in ai_examples.splitlines() if e.strip()]
                if ai_name in st.session_state.intents:
                    st.session_state.intents[ai_name].extend(exs)
                    st.success(f"Appended {len(exs)} examples to `{ai_name}`.")
                else:
                    st.session_state.intents[ai_name] = exs if exs else ["example sentence"]
                    st.success(f"Added new intent `{ai_name}`.")
    st.markdown("---")
    st.download_button(
        label="Save intents.json",
        data=json.dumps(st.session_state.intents, indent=2),
        file_name="intents.json",
        mime="application/json"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div style='background:#0b0d0f;padding:18px;border-radius:8px;margin-bottom:12px;'>", unsafe_allow_html=True)
    st.subheader("NLU Visualizer")
    user_query = st.text_area("User Query", value="Move ₹1500 to account 12345678", height=100)
    top_k = st.number_input("Top intents to show", min_value=1, max_value=10, value=4, step=1)
    if st.button("Analyze"):
        if st.session_state.trained_model and st.session_state.clf and st.session_state.vectorizer:
            vec = st.session_state.vectorizer
            clf = st.session_state.clf
        else:
            vec, clf = train_nlu(st.session_state.intents)
        if clf is None:
            st.warning("No model available. Add examples and train first.")
        else:
            Xq = vec.transform([user_query])

            # Attempt to get raw scores (decision_function) when available
            probs = None
            if hasattr(clf, "decision_function"):
                try:
                    scores = clf.decision_function(Xq)
                    s = scores[0] if scores.ndim > 1 else scores
                    # compute auto temperature from raw scores
                    temp = auto_temperature_from_scores(s)
                    probs = softmax_with_temperature(s, temp)
                except Exception:
                    probs = clf.predict_proba(Xq)[0]
                    # compute margin on probs and sharpen
                    sorted_idxs = np.argsort(probs)[::-1]
                    margin = probs[sorted_idxs[0]] - probs[sorted_idxs[1]] if probs.size > 1 else probs[0]
                    probs = sharpen_probs_by_margin(probs, margin)
            else:
                # fallback: use predict_proba then sharpen based on margin
                raw = clf.predict_proba(Xq)[0]
                sorted_idxs = np.argsort(raw)[::-1]
                margin = raw[sorted_idxs[0]] - raw[sorted_idxs[1]] if raw.size > 1 else raw[0]
                probs = sharpen_probs_by_margin(raw, margin)

            classes = clf.classes_
            idxs = np.argsort(probs)[::-1][:top_k]
            st.markdown("### Intent Recognition")
            for i in idxs:
                prob = clamp_prob(probs[i])
                st.write(f"- **{classes[i]}:** {prob:.2f}")
            st.markdown("### Entity Extraction")
            ents = extract_entities(user_query)
            if ents:
                if "Account_Number" in ents:
                    st.write(f"**Account_Number:** {ents['Account_Number']}")
                if "Amount" in ents:
                    st.write(f"**Amount:** {ents['Amount']:.2f}")
            # else: no entities found; continue
            
            # Action based on intent
            st.markdown("### Action Response")
            if len(idxs) == 0:
                st.warning("No intent predicted.")
            else:
                top_intent = classes[idxs[0]]
                top_prob = clamp_prob(probs[idxs[0]])

                if top_prob > 0.5:
                    if top_intent == "card_block":
                        card_num = ents.get("Account_Number", "****5678")
                        st.success(f"✓ Card ending in {card_num[-4:]} has been **BLOCKED** successfully.")
                    elif top_intent == "transfer_money":
                        amount = ents.get("Amount", 0)
                        account = ents.get("Account_Number", "N/A")
                        if amount > 0:
                            st.success(f"✓ ₹{amount:.2f} transferred to account **{account}** successfully.")
                        else:
                            st.info("Amount not detected. Please specify amount in query.")
                    elif top_intent  == "check_balance":
                        st.info("Your current balance: ₹25,500.00")
                    elif top_intent == "find_atm":
                        st.info("Nearest ATM: 0.5 km away at Main Street Branch")
                    else:
                        st.info(f"Intent: {top_intent}")
                else:
                    st.warning(f"Low confidence ({top_prob:.2f}). Could not process request clearly.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Train model under visualizer................................................
with right_col:
    st.markdown("<div style='background:#0b0d0f;padding:18px;border-radius:8px;'>", unsafe_allow_html=True)
    st.subheader("Train model")
    if st.session_state.trained_model:
        st.markdown('<span style="display:inline-block;padding:6px 10px;background:#d4f5d6;color:#063d10;border-radius:6px;font-weight:600;">Trained model found</span>', unsafe_allow_html=True)
    col_e, col_b, col_l = st.columns([1,1,2])
    with col_e:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=st.session_state.train_params.get("epochs", 2), step=1, key="epochs2")
    with col_b:
        batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=st.session_state.train_params.get("batch_size", 8), step=1, key="batch2")
    with col_l:
        lr = st.number_input("Learning rate", min_value=0.0, format="%.8f", value=float(st.session_state.train_params.get("lr", 0.00002)), step=1e-6, key="lr2")
    if st.button("Start training", key="start_training"):
        st.session_state.train_params = {"epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr)}
        progress = st.progress(0)
        total_steps = max(10, int(epochs) * 5)
        for i in range(total_steps):
            time.sleep(0.03)
            progress.progress(int((i+1)/total_steps * 100))
        try:
            vec, clf = train_nlu(st.session_state.intents)
            if clf is not None:
                st.success("Training completed.")
            else:
                st.warning("No training data available; add intents/examples first.")
        except Exception as e:
            st.error(f"Training failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
