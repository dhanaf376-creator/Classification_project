import streamlit as st
import pickle
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Startup Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("investment_model.pkl", "rb"))
scaler = pickle.load(open("investment_scaler.pkl", "rb"))
market_encoder = pickle.load(open("market_encoder.pkl", "rb"))
country_code = pickle.load(open("country_code.pkl", "rb"))
status = pickle.load(open("status_encoder.pkl", "rb"))

# ---------------- COUNTRY CURRENCY MAP ----------------
currency_map = {

    "ARE": ("AED", "د.إ"),
    "ARG": ("ARS", "$"),
    "ARM": ("AMD", "֏"),
    "AUS": ("AUD", "$"),
    "AUT": ("EUR", "€"),
    "AZE": ("AZN", "₼"),
    "BEL": ("EUR", "€"),
    "BGD": ("BDT", "৳"),
    "BGR": ("BGN", "лв"),
    "BHR": ("BHD", ".د.ب"),
    "BHS": ("BSD", "$"),
    "BLR": ("BYN", "Br"),
    "BMU": ("BMD", "$"),
    "BRA": ("BRL", "R$"),
    "BRN": ("BND", "$"),
    "BWA": ("BWP", "P"),
    "CAN": ("CAD", "$"),
    "CHE": ("CHF", "CHF"),
    "CHL": ("CLP", "$"),
    "CHN": ("CNY", "¥"),
    "CIV": ("XOF", "CFA"),
    "CMR": ("XAF", "FCFA"),
    "COL": ("COP", "$"),
    "CRI": ("CRC", "₡"),
    "CYM": ("KYD", "$"),
    "CYP": ("EUR", "€"),
    "CZE": ("CZK", "Kč"),
    "DEU": ("EUR", "€"),
    "DNK": ("DKK", "kr"),
    "DOM": ("DOP", "RD$"),
    "DZA": ("DZD", "دج"),
    "ECU": ("USD", "$"),
    "EGY": ("EGP", "£"),
    "ESP": ("EUR", "€"),
    "EST": ("EUR", "€"),
    "FIN": ("EUR", "€"),
    "FRA": ("EUR", "€"),
    "GBR": ("GBP", "£"),
    "GHA": ("GHS", "₵"),
    "GIB": ("GIP", "£"),
    "GRC": ("EUR", "€"),
    "GTM": ("GTQ", "Q"),
    "HKG": ("HKD", "$"),
    "HRV": ("EUR", "€"),
    "HUN": ("HUF", "Ft"),
    "IDN": ("IDR", "Rp"),
    "IND": ("INR", "₹"),
    "IRL": ("EUR", "€"),
    "ISR": ("ILS", "₪"),
    "ITA": ("EUR", "€"),
    "JAM": ("JMD", "J$"),
    "JOR": ("JOD", "JD"),
    "JPN": ("JPY", "¥"),
    "KEN": ("KES", "KSh"),
    "KHM": ("KHR", "៛"),
    "KOR": ("KRW", "₩"),
    "KWT": ("KWD", "د.ك"),
    "LAO": ("LAK", "₭"),
    "LBN": ("LBP", "ل.ل"),
    "LIE": ("CHF", "CHF"),
    "LTU": ("EUR", "€"),
    "LUX": ("EUR", "€"),
    "LVA": ("EUR", "€"),
    "MAF": ("EUR", "€"),
    "MAR": ("MAD", "د.م."),
    "MCO": ("EUR", "€"),
    "MDA": ("MDL", "L"),
    "MEX": ("MXN", "$"),
    "MKD": ("MKD", "ден"),
    "MLT": ("EUR", "€"),
    "MMR": ("MMK", "Ks"),
    "MUS": ("MUR", "₨"),
    "MYS": ("MYR", "RM"),
    "NGA": ("NGN", "₦"),
    "NIC": ("NIO", "C$"),
    "NLD": ("EUR", "€"),
    "NOR": ("NOK", "kr"),
    "NPL": ("NPR", "₨"),
    "NZL": ("NZD", "$"),
    "OMN": ("OMR", "ر.ع."),
    "PAK": ("PKR", "₨"),
    "PAN": ("PAB", "B/."),
    "PER": ("PEN", "S/"),
    "PHL": ("PHP", "₱"),
    "POL": ("PLN", "zł"),
    "PRT": ("EUR", "€"),
    "ROM": ("RON", "lei"),
    "RUS": ("RUB", "₽"),
    "SAU": ("SAR", "﷼"),
    "SGP": ("SGD", "$"),
    "SLV": ("USD", "$"),
    "SOM": ("SOS", "S"),
    "SRB": ("RSD", "дин"),
    "SVK": ("EUR", "€"),
    "SVN": ("EUR", "€"),
    "SWE": ("SEK", "kr"),
    "SYC": ("SCR", "₨"),
    "THA": ("THB", "฿"),
    "TTO": ("TTD", "TT$"),
    "TUN": ("TND", "د.ت"),
    "TUR": ("TRY", "₺"),
    "TWN": ("TWD", "NT$"),
    "TZA": ("TZS", "TSh"),
    "UGA": ("UGX", "USh"),
    "UKR": ("UAH", "₴"),
    "URY": ("UYU", "$U"),
    "USA": ("USD", "$"),
    "UZB": ("UZS", "so'm"),
    "VNM": ("VND", "₫"),
    "ZAF": ("ZAR", "R")

}


# ---------------- USD CONVERSION RATES ----------------
usd_rates = {

    "AED": 0.27,
    "ARS": 0.0011,
    "AMD": 0.0026,
    "AUD": 0.66,
    "EUR": 1.08,
    "AZN": 0.59,
    "BDT": 0.0085,
    "BGN": 0.55,
    "BHD": 2.65,
    "BSD": 1.00,
    "BYN": 0.31,
    "BMD": 1.00,
    "BRL": 0.20,
    "BND": 0.74,
    "BWP": 0.073,
    "CAD": 0.74,
    "CHF": 1.12,
    "CLP": 0.0011,
    "CNY": 0.14,
    "XOF": 0.0016,
    "XAF": 0.0016,
    "COP": 0.00025,
    "CRC": 0.0020,
    "KYD": 1.20,
    "CZK": 0.043,
    "DKK": 0.15,
    "DOP": 0.017,
    "DZD": 0.0074,
    "EGP": 0.020,
    "GBP": 1.27,
    "GHS": 0.065,
    "GIP": 1.27,
    "GTQ": 0.13,
    "HKD": 0.13,
    "HUF": 0.0028,
    "IDR": 0.000062,
    "INR": 0.012,
    "ILS": 0.27,
    "JMD": 0.0064,
    "JOD": 1.41,
    "JPY": 0.0067,
    "KES": 0.0077,
    "KHR": 0.00024,
    "KRW": 0.00072,
    "KWD": 3.25,
    "LAK": 0.000046,
    "LBP": 0.000011,
    "MDL": 0.056,
    "MXN": 0.059,
    "MKD": 0.018,
    "MMK": 0.00048,
    "MUR": 0.022,
    "MYR": 0.21,
    "NGN": 0.00066,
    "NIO": 0.027,
    "NOK": 0.095,
    "NPR": 0.0075,
    "NZD": 0.61,
    "OMR": 2.60,
    "PKR": 0.0036,
    "PAB": 1.00,
    "PEN": 0.27,
    "PHP": 0.017,
    "PLN": 0.25,
    "RON": 0.22,
    "RUB": 0.011,
    "SAR": 0.27,
    "SGD": 0.74,
    "SOS": 0.0017,
    "RSD": 0.0092,
    "SEK": 0.094,
    "SCR": 0.073,
    "THB": 0.027,
    "TTD": 0.15,
    "TND": 0.32,
    "TRY": 0.031,
    "TWD": 0.031,
    "TZS": 0.00039,
    "UGX": 0.00027,
    "UAH": 0.025,
    "UYU": 0.026,
    "USD": 1.00,
    "UZS": 0.000079,
    "VND": 0.000039,
    "ZAR": 0.054

}

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

header {
    visibility: hidden;
}

.block-container {
    padding-top: 0rem;
}

.stApp {
    background: linear-gradient(135deg,#050b1f,#0a1633,#020814);
    color: white;
}

.big-title {
    font-size: 50px;
    font-weight: 800;
    color: white;
    text-align:center;
    margin-top: 25px;
}

.subtitle{
    font-size:14px;
    color:#8fa7d6;
    text-align:center;
    margin-top:-6px;
    margin-bottom:25px;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 12px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 15px rgba(0,255,255,0.06);
    height: 90px;
    margin-top: -17px;
}

.metric-card h2 {
    font-size: 20px;
    margin-top: -17px;
}

.metric-card h1 {
    font-size: 20px;
    margin-top: -25px;
}

.prediction-card {
    background: rgba(255,255,255,0.05);
    padding: 40px;
    border-radius: 25px;
    text-align: center;
    border: 1px solid rgba(0,255,255,0.2);
    box-shadow: 0 0 35px rgba(0,255,255,0.15);
    min-height: 110px;
    margin-top: 20px;
}

.stButton > button {
    width:100%;
    background: linear-gradient(90deg, #1f6fff 0%, #8e2de2 100%) !important;
    color:white !important;
    border:none !important;
    border-radius:18px !important;
    height:60px !important;
    font-size:18px !important;
    font-weight:600 !important;
    box-shadow:none !important;
    margin-top: -15px !important
}

            
.stButton > button:hover {
    background: linear-gradient(90deg, #2d7dff 0%, #9d3df0 100%) !important;
    color:white !important;
    border:none !important;
}

label {
    color: white !important;
    font-weight: 600 !important;
}

.stNumberInput,
.stSelectbox {
    width: 80% !important;
}

[data-testid="stHorizontalBlock"] {
    margin-top: -18px;
}

label {
    color: white !important;
    font-weight: 600 !important;
    margin-top: -3px !important;
    margin-bottom: 2px !important;
}

.stNumberInput,
.stSelectbox {
    width: 80% !important;
    margin-top: -3px !important;
    margin-bottom: 3px !important;
}    

.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: white !important;
}

div[role="listbox"] {
    background-color: #0b1228 !important;
    border-radius: 10px !important;
}

div[role="option"] {
    background-color: #0b1228 !important;
    color: white !important;
}

div[role="option"]:hover {
    background-color: rgba(255,255,255,0.08) !important;
}

div[data-baseweb="select"] > div {
    color: white !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

            
.stNumberInput div[data-baseweb="input"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}


.stNumberInput input {
    background: transparent !important;
    color: white !important;
    border: none !important;
}

.stNumberInput button {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border: none !important;
}

.stNumberInput div {
    background: transparent !important;
}

.stTextInput div[data-baseweb="input"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
}

.stTextInput input {
    color: white !important;
    background: transparent !important;
}

::placeholder {
    color: rgba(255,255,255,0.5) !important;
}

</style>
""", unsafe_allow_html=True)
 

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("""
    <style>

    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg,#0b1228,#111c38);
        border-right:1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] *{
        border-color: transparent !important;
    }

    section[data-testid="stSidebar"] .block-container{
        background: transparent !important;
        padding-top: 0rem !important;
    }

    section[data-testid="stSidebar"] .element-container{
        background: transparent !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{
        background: transparent !important;
    }


    .logo-box{
        text-align:center;
        padding:5px 10px 10px;
        margin-top:-70px;
        margin-bottom:15px;
    }

    .logo-box img{
        width:90px;
        margin-bottom:2px;
    }

    .brand{
        font-size:32px;
        font-weight:800;
        color:white;
    }

    .brand span{
        color:#00c8ff;
    }

    .subtext{
        color:#8fa7d6;
        font-size:14px;
        margin-top:8px; 
    } 
                

   .menu-container{
       display:flex;
       flex-direction:column;
       gap:0px !important;

       margin-top:-145px;
       margin-bottom:-45px;
       padding-top:0 !important;
       padding-bottom:0 !important;
}


    section[data-testid="stSidebar"] div.stButton{
       margin:-18px 0 !important;
       padding:0 !important;
}

     div.stButton > button{
       width:260px !important;
       min-width:260px !important;
       height:52px !important;

       margin:-4px 0 !important;
       padding:0 22px !important;

       border-radius:10px !important;
       border:none !important;

       background: rgba(255,255,255,0.05) !important;
       color:white !important;

       font-size:13px !important;
       font-weight:500 !important;

       text-align:left !important;

       transition:none !important;

       box-shadow:none !important;

       transform:none !important;
}


    div.stButton > button:hover{
       background: rgba(255,255,255,0.10) !important;
       color:white !important;

       transform:none !important;
}

    .active-btn button{
       background: linear-gradient(90deg,#1f6fff,#8e2de2) !important;
       color:white !important;

       height:55px !important;
       margin:2px 0 !important;
       padding:0 18px !important;

       border:none !important;
       box-shadow:none !important;

       transform:none !important;
}
    

    div[data-testid="stSidebar"] button[kind="secondary"]{
        background: transparent !important;
        border: none !important;
        color: white !important;
        font-size: 28px !important;
        width: 42px !important;
        height: 42px !important;
        padding: 0 !important;
        margin-top: -75px !important;
        margin-left: -8px !important;
        box-shadow: none !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # ---------------- LOGO ----------------

    st.markdown("""
    <div class="logo-box">
        <img src="https://cdn-icons-png.flaticon.com/512/3212/3212608.png">
        <div class="brand">Startup<span>AI</span></div>
        <div class="subtext">Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
    if "selected_page" not in st.session_state:
      st.session_state.selected_page = "Dashboard"

    if "prediction_result" not in st.session_state:
      st.session_state.prediction_result = None

    # ---------------- MENU BUTTONS ----------------

    st.markdown('<div class="menu-container">', unsafe_allow_html=True)

    # DASHBOARD

    dashboard_class = (
        "active-btn"
        if st.session_state.selected_page == "Dashboard"
        else ""
    )

    st.markdown(
        f'<div class="{dashboard_class}">',
        unsafe_allow_html=True
    )

    if st.button("🏠 Dashboard", key="dashboard_btn"):
        st.session_state.selected_page = "Dashboard"

    st.markdown('</div>', unsafe_allow_html=True)

    # MODEL INSIGHTS

    insights_class = (
        "active-btn"
        if st.session_state.selected_page == "Model Insights"
        else ""
    )

    st.markdown(
        f'<div class="{insights_class}">',
        unsafe_allow_html=True
    )

    if st.button("📊 Model Insights", key="insights_btn"):
        st.session_state.selected_page = "Model Insights"

    st.markdown('</div>', unsafe_allow_html=True)

    # AI RECOMMENDATIONS

    ai_class = (
        "active-btn"
        if st.session_state.selected_page == "AI Recommendations"
        else ""
    )

    st.markdown(
        f'<div class="{ai_class}">',
        unsafe_allow_html=True
    )

    if st.button("💡 AI Recommendations", key="ai_btn"):
        st.session_state.selected_page = "AI Recommendations"

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- SELECTED PAGE ----------------

    selected = st.session_state.selected_page

    # ---------------- ABOUT CARD ----------------

    st.markdown("""
    <div style="
    background:rgba(255,255,255,0.05);
    padding:16px;
    border-radius:10px;
    border:1px solid rgba(255,255,255,0.08);
    margin-top:-18px;
    margin-bottom:25px;
    box-shadow:0 0 18px rgba(0,255,255,0.05);
    ">

    <div style="
    color:white;
    font-size:16px;
    font-weight:700;
    margin-bottom:10px;
    ">
    🤖 About Dashboard
    </div>

    <div style="
    color:#cbd5e1;
    font-size:13px;
    line-height:1.8;
    ">

    AI-powered startup analytics platform for predicting startup success using funding, market, and investment insights.

    

    ✔ ML Prediction<br>
    ✔ Investment Analysis<br>
    ✔ AI Recommendations

    </div>

    </div>
    """, unsafe_allow_html=True)

    # ---------------- REFRESH ----------------

    if st.button("🔄", key="refresh_btn"):

        st.session_state.market = None
        st.session_state.country = None

        st.session_state.funding_total = 0.0
        st.session_state.funding_rounds = 1

        st.session_state.seed = 0.0
        st.session_state.venture = 0.0
        st.session_state.debt_financing = 0.0
        st.session_state.angel = 0.0

        st.rerun()


# DASHBOARD PAGE
if selected == "Dashboard":

    # ---------------- HEADER ----------------

    st.markdown(
        '<div class="big-title">🚀 AI Startup Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Predict • Analyze • Invest Smarter</div>',
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # INPUT SECTION
    st.markdown("""
    <style>

    /* Dropdown box */
    div[data-baseweb="select"] > div {
        background-color: #18264a !important;
        border: 1px solid #2d3b5f !important;
        border-radius: 14px !important;
        color: white !important;
    }

    /* Selected text + placeholder */
    div[data-baseweb="select"] span {
        color: white !important;
    }

    /* Placeholder */
    div[data-baseweb="select"] input::placeholder {
        color: white !important;
        opacity: 1 !important;
    }

    /* Force placeholder visible in white */
    div[data-baseweb="select"] div {
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)

    input_section, result_section = st.columns([3, 1])

  
    # INPUT COLUMN
    with input_section:

        st.markdown("""
        <h2 style='color:white;
        font-size:22px;
        margin-top:-4px;
        margin-bottom:19px;'>
        📋 Startup Details
        </h2>
        """, unsafe_allow_html=True)

        market_list = list(market_encoder.classes_)

        country_list = [
            str(i) for i in country_code.classes_
            if str(i) != "nan"
        ]

        col1, col2 = st.columns(2)

        # ---------------- COLUMN 1 ----------------

        with col1:

            market = st.selectbox(
                "📊 Market",
                market_list,
                index=None,
                placeholder="Select",
                key="market"
            )

            country = st.selectbox(
                "🏳️ Country Code",
                sorted(country_list),
                index=None,
                placeholder="Select",
                key="country"
            )

            currency_name, currency_symbol = currency_map.get(
                country,
                ("USD", "$")
            )

            if country is None:
                currency_name = "USD"
                currency_symbol = "$"

            funding_total = st.number_input(
                f"💰 Funding Total ({currency_symbol})",
                min_value=0.0,
                value=0.0,
                key="funding_total"
            )

            funding_rounds = st.number_input(
                "📈 Funding Rounds",
                min_value=0.0,
                value=0.0,
                key="funding_rounds"
            )

        # ---------------- COLUMN 2 ----------------

        with col2:

            seed = st.number_input(
                f"🌱 Seed Funding ({currency_symbol})",
                min_value=0.0,
                value=0.0,
                key="seed"
            )

            venture = st.number_input(
                f"🚀 Venture Funding ({currency_symbol})",
                min_value=0.0,
                value=0.0,
                key="venture"
            )

            debt_financing = st.number_input(
                f"🏦 Debt Financing ({currency_symbol})",
                min_value=0.0,
                value=0.0,
                key="debt_financing"
            )

            angel = st.number_input(
                f"👼 Angel Funding ({currency_symbol})",
                min_value=0.0,
                value=0.0,
                key="angel"
            )

        st.write("")

        predict = st.button("🚀 Predict Startup Success")

    # RESULT SECTION
    with result_section:

        st.markdown("""
        <div class="prediction-card" style="margin-top:65px;">
        """, unsafe_allow_html=True)

        st.markdown(
            """
            <h2 style='
            color:white;
            font-size:22px;
            margin-top:-70px;
            margin-bottom:5px;
            text-align:left;
            padding-left:25px;
            '>
            🤖 Prediction Result
            </h2>
            """,
            unsafe_allow_html=True
        )

        if predict:

            country = str(country)

            market_encoded = market_encoder.transform([market])[0]
            country_encoded = country_code.transform([country])[0]

            # ---------------- CONVERT TO USD ----------------

            rate = usd_rates.get(currency_name, 1)

            funding_total_usd = funding_total * rate
            seed_usd = seed * rate
            venture_usd = venture * rate
            debt_financing_usd = debt_financing * rate
            angel_usd = angel * rate

            # ---------------- INPUT ARRAY ----------------

            input_data = np.array([[
                market_encoded,
                funding_total_usd,
                country_encoded,
                funding_rounds,
                seed_usd,
                venture_usd,
                debt_financing_usd,
                angel_usd
            ]])

            # ---------------- SCALE ----------------

            input_scaled = scaler.transform(input_data)

            # ---------------- PREDICT ----------------

            prediction = model.predict(input_scaled)[0]

            # STATUS PREDICTION
            predicted_status = status.inverse_transform([prediction])[0]

            # SAVE RESULT FOR AI RECOMMENDATIONS PAGE
            st.session_state.prediction_result = predicted_status

            if predicted_status == "operating":

                result = "OPERATING 🚀"
                color = "#1b9d69"

            elif predicted_status == "acquired":

                result = "ACQUIRED 🏆"
                color = "#1dc2e3"

            else:

                result = "CLOSED ❌"
                color = "#B71D1D"

            # ---------------- RESULT ----------------
            st.markdown(
                f"""
                <h1 style='
                color:{color};
                font-size:24px;
                margin-top:-130px;
                width:100%;
                text-align:center;
                '>
                {result}
                </h1>
                """,
                unsafe_allow_html=True
            )
                # ---------------- INFO ----------------
        if predict:

            st.markdown(
                f"""
                <div style="
                background:rgba(255,255,255,0.05);
                padding:8px 12px;
                border-radius:10px;
                margin-top:-15px;
                margin-bottom:5px;
                font-size:13px;
                color:white;
                border:1px solid rgba(255,255,255,0.08);
                ">
                📊 <b>Market:</b> {market}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                background:rgba(255,255,255,0.05);
                padding:8px 12px;
                border-radius:10px;
                margin-bottom:5px;
                font-size:13px;
                color:white;
                border:1px solid rgba(255,255,255,0.08);
                ">
                🏳️ <b>Country:</b> {country}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                background:rgba(255,255,255,0.05);
                padding:8px 12px;
                border-radius:10px;
                margin-bottom:5px;
                font-size:13px;
                color:white;
                border:1px solid rgba(255,255,255,0.08);
                ">
                💱 <b>Currency:</b> {currency_name}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                background:rgba(255,255,255,0.05);
                padding:8px 12px;
                border-radius:10px;
                margin-bottom:5px;
                font-size:13px;
                color:white;
                border:1px solid rgba(255,255,255,0.08);
                ">
                💰 <b>Funding:</b> {currency_symbol}{funding_total:,.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

# MODEEL INSIGHTS PAG
elif selected == "Model Insights":

    # ---------------- HEADER ----------------
    st.markdown(
        '<div class="big-title">📊 Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Understanding Model Performance & Startup Analytics</div>',
        unsafe_allow_html=True
    )

    st.write("")


    # ABOUT MODEL
    st.markdown("""
    <div style="
    background:rgba(255,255,255,0.05);
    padding:24px;
    border-radius:22px;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 0 20px rgba(0,255,255,0.06);
    margin-bottom:18px;
    ">

    <h2 style="
    color:white;
    font-size:24px;
    margin-top:0px;
    margin-bottom:14px;
    ">
    🤖 About The Prediction Model
    </h2>

    <div style="
    color:#cbd5e1;
    line-height:1.8;
    font-size:15px;
    ">

    This dashboard uses an <b>XGBoost Machine Learning Model</b>
    trained on startup investment and funding datasets.

    <br>

    The model predicts startup outcomes using financial,
    investment, and market-related features.

    <br>

    <b>Possible Prediction Results:</b>

    <div style="margin-top:10px; line-height:1.9;">
    🚀 <b>Operating</b> — Startup is actively running and growing.<br>
    🏆 <b>Acquired</b> — Startup shows strong acquisition potential.<br>
    ❌ <b>Closed</b> — Startup has higher operational risk.
    </div>

    <br>

    The system evaluates multiple business indicators to estimate
    startup success probability and operational sustainability.

    </div>

    </div>
    """, unsafe_allow_html=True)

    # KEY FACTORS
    st.markdown("""
    <div style="
    background:rgba(255,255,255,0.05);
    padding:24px;
    border-radius:22px;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:0 0 20px rgba(0,255,255,0.06);
    ">

    <h2 style="
    color:white;
    font-size:24px;
    margin-top:0px;
    margin-bottom:16px;
    ">
    🔍 Key Factors Influencing Predictions
    </h2>

    <div style="
    color:#cbd5e1;
    line-height:1.9;
    font-size:15px;
    ">

    💰 <b>Total Funding</b> — Indicates overall investor confidence and business scale.<br>

    🌱 <b>Seed Funding</b> — Reflects early-stage startup support and market validation.<br>

    🚀 <b>Venture Capital</b> — Represents long-term growth and scalability potential.<br>

    📈 <b>Funding Rounds</b> — Measures investment consistency and business progress.<br>

    🌍 <b>Market & Country</b> — Represents industry demand, competition, and economic opportunities.<br>

    🏦 <b>Debt Financing</b> — Helps evaluate financial stability and liabilities.<br>

    👼 <b>Angel Investment</b> — Indicates trust from early investors and startup attractiveness.

    </div>

    </div>
    """, unsafe_allow_html=True)


# AI RECOMMENDATIONS PAGE
elif selected == "AI Recommendations":

    # ---------------- HEADER ----------------
    st.markdown(
        '<div class="big-title">💡 AI Recommendations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Smart Suggestions Based On Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # GET PREDICTION STATUS
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    result = st.session_state.prediction_result

  
    # NO PREDICTION YET
    if result is None:

        st.markdown("""
        <div style="
        background:rgba(255,255,255,0.05);
        padding:30px;
        border-radius:22px;
        border:1px solid rgba(255,255,255,0.08);
        text-align:center;
        color:white;
        font-size:18px;
        ">

        🚀 Please predict a startup first to get AI recommendations.

        </div>
        """, unsafe_allow_html=True)


    # OPERATING STARTUP
    elif result == "operating":

        st.markdown("""
        <div style="
        background:rgba(255,255,255,0.05);
        padding:28px;
        border-radius:22px;
        border:1px solid rgba(0,255,120,0.15);
        box-shadow:0 0 20px rgba(0,255,120,0.08);
        ">

        <h2 style="
        color:#00ff99;
        margin-top:0;
        margin-bottom:18px;
        ">
        🚀 Startup Status: OPERATING
        </h2>

        <div style="
        color:#cbd5e1;
        line-height:1.9;
        font-size:15px;
        ">

        ✔ Continue scaling business operations strategically.<br><br>

        ✔ Increase market expansion and customer acquisition.<br><br>

        ✔ Strengthen investor relationships for future funding rounds.<br><br>

        ✔ Invest more in innovation and product development.<br><br>

        ✔ Focus on operational efficiency and revenue growth.<br><br>

        ✔ Use analytics and AI tools for smarter decision-making.

        </div>

        </div>
        """, unsafe_allow_html=True)

    # ACQUIRED STARTUP
    elif result == "acquired":

        st.markdown("""
        <div style="
        background:rgba(255,255,255,0.05);
        padding:28px;
        border-radius:22px;
        border:1px solid rgba(0,200,255,0.15);
        box-shadow:0 0 20px rgba(0,200,255,0.08);
        ">

        <h2 style="
        color:#00d4ff;
        margin-top:0;
        margin-bottom:18px;
        ">
        🏆 Startup Status: ACQUIRED
        </h2>

        <div style="
        color:#cbd5e1;
        line-height:1.9;
        font-size:15px;
        ">

        ✔ Focus on maintaining strong business value.<br><br>

        ✔ Improve scalability and long-term sustainability.<br><br>

        ✔ Enhance product quality and customer retention.<br><br>

        ✔ Build strategic partnerships and collaborations.<br><br>

        ✔ Optimize financial management and operational structure.<br><br>

        ✔ Increase innovation to remain competitive in the market.

        </div>

        </div>
        """, unsafe_allow_html=True)

    # CLOSED STARTUP
    elif result == "closed":

        st.markdown("""
        <div style="
        background:rgba(255,255,255,0.05);
        padding:28px;
        border-radius:22px;
        border:1px solid rgba(255,0,80,0.15);
        box-shadow:0 0 20px rgba(255,0,80,0.08);
        ">

        <h2 style="
        color:#ff4d6d;
        margin-top:0;
        margin-bottom:18px;
        ">
        ❌ Startup Status: CLOSED
        </h2>

        <div style="
        color:#cbd5e1;
        line-height:1.9;
        font-size:15px;
        ">

        ✔ Improve funding strategy and investor confidence.<br><br>

        ✔ Reduce operational risks and unnecessary expenses.<br><br>

        ✔ Focus on achieving strong product-market fit before scaling.<br><br>

        ✔ Increase marketing and customer engagement efforts.<br><br>

        ✔ Build stronger financial planning and cash-flow management.<br><br>

        ✔ Analyze competitor trends and market demand carefully.

        </div>

        </div>
        """, unsafe_allow_html=True)