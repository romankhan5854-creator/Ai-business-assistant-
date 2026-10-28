"""AI Business Assistant - helps small shops track sales, write ads,
organize expenses and answer customer questions. Includes simple Urdu voice input.

Run locally:  streamlit run app.py
"""

import json
import os
import re
from datetime import date

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-flash"

# Demo monthly plans (no real payment is processed in this version).
PLANS = {
    "Free": {"price": "Rs 0 / month", "limit": 10, "perks": "10 AI requests, all basic tools"},
    "Starter": {"price": "Rs 999 / month", "limit": 150, "perks": "150 AI requests, Urdu voice input"},
    "Pro": {"price": "Rs 2,499 / month", "limit": 1000, "perks": "1000 AI requests, priority support"},
}

EXPENSE_CATEGORIES = ["Rent", "Utilities", "Stock / Inventory", "Salaries", "Transport", "Marketing", "Other"]

SALES_COLS = ["Date", "Item", "Quantity", "Unit Price", "Total"]
EXP_COLS = ["Date", "Category", "Description", "Amount"]

st.set_page_config(page_title="AI Business Assistant", page_icon="🏪", layout="wide")


# ----------------------------------------------------------------------------
# Helpers (pure logic - easy to test)
# ----------------------------------------------------------------------------
def parse_json(text: str):
    """Extract a JSON object from model output, tolerating ```json fences."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def to_number(value, default=0.0) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def make_sale_row(item, qty, price, when=None) -> dict:
    qty, price = to_number(qty, 1), to_number(price)
    return {
        "Date": when or date.today().isoformat(),
        "Item": str(item).strip() or "Unknown item",
        "Quantity": qty,
        "Unit Price": price,
        "Total": round(qty * price, 2),
    }


def make_expense_row(category, description, amount, when=None) -> dict:
    return {
        "Date": when or date.today().isoformat(),
        "Category": category if category in EXPENSE_CATEGORIES else "Other",
        "Description": str(description).strip(),
        "Amount": to_number(amount),
    }


def summarize(sales: pd.DataFrame, expenses: pd.DataFrame) -> dict:
    revenue = float(sales["Total"].sum()) if not sales.empty else 0.0
    spent = float(expenses["Amount"].sum()) if not expenses.empty else 0.0
    return {"revenue": revenue, "expenses": spent, "profit": revenue - spent}


def quota_left() -> int:
    plan = PLANS[st.session_state.plan]
    return plan["limit"] - st.session_state.ai_calls


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "") or st.session_state.get("manual_key", "")


def get_model() -> str:
    try:
        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def call_gemini(contents, system: str, as_json: bool = False):
    """Single place where we talk to Gemini. Returns text or None (error shown in UI)."""
    key = get_api_key()
    if not key:
        st.error("Add your Gemini API key in the sidebar (or in Streamlit secrets) first.")
        return None
    if quota_left() <= 0:
        st.warning("You have used all AI requests for your plan. Upgrade in the Plans tab.")
        return None
    try:
        client = genai.Client(api_key=key)
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.7,
            response_mime_type="application/json" if as_json else "text/plain",
        )
        response = client.models.generate_content(model=get_model(), contents=contents, config=config)
        st.session_state.ai_calls += 1
        return response.text
    except Exception as exc:  # network, bad key, quota, safety block...
        st.error(f"Gemini error: {exc}")
        return None


def shop_context() -> str:
    p = st.session_state.profile
    return (
        f"Shop name: {p['name'] or 'not given'}. Type: {p['type'] or 'not given'}. "
        f"City: {p['city'] or 'not given'}. Currency: PKR."
    )


BASE_SYSTEM = (
    "You are an AI Business Assistant for small shop owners and freelancers in Pakistan. "
    "Be practical, short and friendly. Use simple words. "
)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "sales": pd.DataFrame(columns=SALES_COLS),
        "expenses": pd.DataFrame(columns=EXP_COLS),
        "ai_calls": 0,
        "plan": "Free",
        "profile": {"name": "", "type": "", "city": ""},
        "faq_info": "",
        "voice_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_row(key: str, row: dict):
    st.session_state[key] = pd.concat([st.session_state[key], pd.DataFrame([row])], ignore_index=True)


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.title("🏪 AI Business Assistant")
        st.caption("For shop owners & freelancers")

        if not (os.getenv("GEMINI_API_KEY") or _secret_has_key()):
            st.text_input("Gemini API key", type="password", key="manual_key",
                          help="Get a free key at aistudio.google.com")

        st.subheader("Your shop")
        p = st.session_state.profile
        p["name"] = st.text_input("Shop / business name", p["name"])
        p["type"] = st.text_input("What do you sell?", p["type"], placeholder="e.g. clothes, mobile accessories")
        p["city"] = st.text_input("City", p["city"])

        st.divider()
        plan = PLANS[st.session_state.plan]
        st.write(f"**Plan:** {st.session_state.plan} ({plan['price']})")
        st.progress(min(st.session_state.ai_calls / plan["limit"], 1.0))
        st.caption(f"{st.session_state.ai_calls} / {plan['limit']} AI requests used")


def _secret_has_key() -> bool:
    try:
        return "GEMINI_API_KEY" in st.secrets
    except Exception:
        return False


def tab_dashboard():
    s = summarize(st.session_state.sales, st.session_state.expenses)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total sales", f"Rs {s['revenue']:,.0f}")
    c2.metric("Total expenses", f"Rs {s['expenses']:,.0f}")
    c3.metric("Profit", f"Rs {s['profit']:,.0f}")

    if not st.session_state.sales.empty:
        st.subheader("Sales by item")
        st.bar_chart(st.session_state.sales.groupby("Item")["Total"].sum())

    if st.button("💡 Get AI business tips", disabled=st.session_state.sales.empty and st.session_state.expenses.empty):
        prompt = (
            f"{shop_context()}\nSales data:\n{st.session_state.sales.to_csv(index=False)}\n"
            f"Expenses data:\n{st.session_state.expenses.to_csv(index=False)}\n"
            "Give: 1) a 2-line summary, 2) best selling items, 3) where money is leaking, "
            "4) three clear actions for this week. Keep it short."
        )
        with st.spinner("Thinking..."):
            out = call_gemini(prompt, BASE_SYSTEM)
        if out:
            st.markdown(out)


def tab_sales():
    st.subheader("📈 Track sales")
    with st.form("sale_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        item = c1.text_input("Item")
        qty = c2.number_input("Quantity", min_value=1.0, value=1.0, step=1.0)
        price = c3.number_input("Unit price (Rs)", min_value=0.0, value=0.0, step=10.0)
        if st.form_submit_button("➕ Add sale") and item.strip():
            add_row("sales", make_sale_row(item, qty, price))
            st.rerun()

    st.dataframe(st.session_state.sales, use_container_width=True, hide_index=True)
    _csv_tools("sales", SALES_COLS)


def tab_expenses():
    st.subheader("🧾 Organize expenses")
    with st.form("exp_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 3, 1])
        cat = c1.selectbox("Category", EXPENSE_CATEGORIES)
        desc = c2.text_input("Description")
        amt = c3.number_input("Amount (Rs)", min_value=0.0, value=0.0, step=100.0)
        if st.form_submit_button("➕ Add expense") and amt > 0:
            add_row("expenses", make_expense_row(cat, desc, amt))
            st.rerun()

    df = st.session_state.expenses
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        st.bar_chart(df.groupby("Category")["Amount"].sum())
    _csv_tools("expenses", EXP_COLS)


def _csv_tools(key: str, cols: list):
    """Download / upload so data survives restarts (Streamlit Cloud storage is temporary)."""
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV", st.session_state[key].to_csv(index=False),
                       file_name=f"{key}.csv", mime="text/csv", key=f"dl_{key}")
    up = c2.file_uploader("Load saved CSV", type="csv", key=f"up_{key}")
    # file_uploader keeps its file on every rerun, so load each upload only once;
    # otherwise it would overwrite rows added after loading.
    if up is not None and st.session_state.get(f"loaded_{key}") != getattr(up, "file_id", up.name):
        try:
            df = pd.read_csv(up)
            if all(c in df.columns for c in cols):
                st.session_state[key] = df[cols].reset_index(drop=True)
                st.session_state[f"loaded_{key}"] = getattr(up, "file_id", up.name)
                st.success("Data loaded.")
                st.rerun()
            else:
                st.error(f"CSV must have columns: {', '.join(cols)}")
        except pd.errors.ParserError as exc:
            st.error(f"Could not read CSV: {exc}")


def tab_ads():
    st.subheader("📢 Write an advertisement")
    c1, c2 = st.columns(2)
    product = c1.text_input("Product / service", placeholder="e.g. Summer lawn suits")
    offer = c2.text_input("Offer (optional)", placeholder="e.g. 20% off until Friday")
    c3, c4, c5 = st.columns(3)
    platform = c3.selectbox("Where will you post?", ["WhatsApp status", "Facebook post", "Instagram caption", "SMS", "Shop poster"])
    tone = c4.selectbox("Tone", ["Friendly", "Professional", "Exciting", "Festive"])
    lang = c5.selectbox("Language", ["Urdu", "Roman Urdu", "English"])

    if st.button("✨ Generate ad", disabled=not product.strip()):
        prompt = (
            f"{shop_context()}\nWrite 3 different short ads for: {product}. Offer: {offer or 'none'}. "
            f"Platform: {platform}. Tone: {tone}. Language: {lang}. "
            "Include a call to action and 3-5 relevant hashtags only if the platform needs them."
        )
        with st.spinner("Writing..."):
            out = call_gemini(prompt, BASE_SYSTEM)
        if out:
            st.markdown(out)


def tab_faq():
    st.subheader("💬 Answer customer questions")
    st.session_state.faq_info = st.text_area(
        "Your shop info (timings, delivery, return policy, payment methods, prices)",
        st.session_state.faq_info, height=140,
        placeholder="Open 10am-9pm. Delivery Rs 150 in Peshawar. Returns within 7 days with receipt. JazzCash/EasyPaisa accepted.",
    )
    question = st.text_area("Customer's message", placeholder="Paste what the customer asked (any language)")
    c1, c2 = st.columns(2)

    if c1.button("🤖 Draft reply", disabled=not question.strip()):
        prompt = (
            f"{shop_context()}\nShop info:\n{st.session_state.faq_info or 'none provided'}\n\n"
            f"Customer message: {question}\n\n"
            "Write a polite, short reply in the SAME language as the customer. "
            "Use ONLY the shop info above for facts. If the answer is not in the info, say you will confirm and get back."
        )
        with st.spinner("Writing reply..."):
            out = call_gemini(prompt, BASE_SYSTEM)
        if out:
            st.markdown("**Suggested reply**")
            st.code(out, language=None)

    if c2.button("📋 Make FAQ list", disabled=not st.session_state.faq_info.strip()):
        prompt = (
            f"{shop_context()}\nShop info:\n{st.session_state.faq_info}\n\n"
            "Create 8 common customer questions with short answers, in English and Urdu. Use only the info given."
        )
        with st.spinner("Creating FAQ..."):
            out = call_gemini(prompt, BASE_SYSTEM)
        if out:
            st.markdown(out)


VOICE_SYSTEM = (
    "You listen to short voice notes from a shopkeeper in Urdu (or Urdu mixed with English/Punjabi/Pashto). "
    "Transcribe them and extract ONE action. Reply ONLY as JSON with keys: "
    '"transcript" (Urdu text), "english" (English translation), '
    '"intent" (one of "sale", "expense", "question", "unknown"), '
    '"item" (string), "quantity" (number), "amount" (number: unit price for a sale, total for an expense), '
    '"category" (one of ' + ", ".join(EXPENSE_CATEGORIES) + " for expenses, else empty), "
    '"answer" (a short Urdu answer if intent is "question", else empty). '
    "Convert spoken numbers (e.g. 'paanch sau') to digits."
)


def tab_voice():
    st.subheader("🎙️ Urdu voice input")
    st.caption("Say something like: «آج پانچ قمیض آٹھ سو روپے کی بیچیں» or «بجلی کا بل تین ہزار»")
    audio = st.audio_input("Record your voice note")

    if audio is not None and st.button("🔊 Understand my voice"):
        part = types.Part.from_bytes(data=audio.getvalue(), mime_type=audio.type or "audio/wav")
        with st.spinner("Listening..."):
            out = call_gemini([part, "Process this voice note."], VOICE_SYSTEM, as_json=True)
        data = parse_json(out) if out else None
        if data:
            st.session_state.voice_result = data
        elif out:
            st.error("Could not understand the response. Please record again.")

    res = st.session_state.voice_result
    if not res:
        return

    st.markdown(f"**I heard:** {res.get('transcript', '')}")
    st.caption(res.get("english", ""))
    intent = res.get("intent", "unknown")

    if intent == "sale":
        row = make_sale_row(res.get("item", ""), res.get("quantity", 1), res.get("amount", 0))
        st.info(f"Sale: {row['Quantity']:g} × {row['Item']} @ Rs {row['Unit Price']:,.0f} = Rs {row['Total']:,.0f}")
        if st.button("✅ Save this sale"):
            add_row("sales", row)
            st.session_state.voice_result = None
            st.rerun()
    elif intent == "expense":
        row = make_expense_row(res.get("category", "Other"), res.get("item", ""), res.get("amount", 0))
        st.info(f"Expense: {row['Category']} – {row['Description']} – Rs {row['Amount']:,.0f}")
        if st.button("✅ Save this expense"):
            add_row("expenses", row)
            st.session_state.voice_result = None
            st.rerun()
    elif intent == "question":
        st.success(res.get("answer") or "No answer available.")
    else:
        st.warning("I could not find a sale or expense in that. Please try again.")


def tab_plans():
    st.subheader("💳 Monthly business plans")
    cols = st.columns(len(PLANS))
    for col, (name, plan) in zip(cols, PLANS.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"### {name}")
                st.markdown(f"**{plan['price']}**")
                st.write(plan["perks"])
                current = st.session_state.plan == name
                if st.button("Current plan" if current else f"Choose {name}", key=f"plan_{name}", disabled=current):
                    st.session_state.plan = name
                    st.rerun()
    st.caption("Demo only: switching plans changes the AI request limit but no payment is taken. "
               "Connect a payment provider (e.g. Stripe, JazzCash) before charging real customers.")


def main():
    init_state()
    sidebar()
    st.title(f"🏪 {st.session_state.profile['name'] or 'AI Business Assistant'}")
    tabs = st.tabs(["📊 Dashboard", "🎙️ Voice (Urdu)", "📈 Sales", "🧾 Expenses", "📢 Ads", "💬 Customers", "💳 Plans"])
    for tab, fn in zip(tabs, [tab_dashboard, tab_voice, tab_sales, tab_expenses, tab_ads, tab_faq, tab_plans]):
        with tab:
            fn()


if __name__ == "__main__":
    main()
