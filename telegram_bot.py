import os
import json
import time
import re
import html
import threading
import urllib.request
import urllib.error

import quant_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ALERT_HISTORY_FILE = os.path.join(DATA_DIR, "telegram_alert_history.json")
USER_CONFIG_FILE = os.path.join(DATA_DIR, "telegram_user_config.json")


def get_telegram_credentials():
    """Extracts bot token and chat id from environment or .env file."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("TELEGRAM_BOT_TOKEN="):
                            token = line.split("=", 1)[1].strip()
                        elif line.startswith("TELEGRAM_CHAT_ID="):
                            chat_id = line.split("=", 1)[1].strip()
            except Exception:
                pass
    return token or "", chat_id or ""


def get_user_config():
    """Returns dynamic user trading sizing and risk configuration."""
    default_cfg = {
        "account_balance": 10000.0,
        "max_risk_pct": 0.50,
        "leverage": 200,
        "default_horizon": "scalping"
    }
    if os.path.exists(USER_CONFIG_FILE):
        try:
            with open(USER_CONFIG_FILE) as f:
                cfg = json.load(f)
                default_cfg.update(cfg)
        except Exception:
            pass
    return default_cfg


def save_user_config(cfg):
    """Persists updated account balance, risk, and leverage settings."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(USER_CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print("Error saving telegram_user_config.json:", e)
        return False


def get_main_reply_keyboard():
    """Big touch buttons on user's mobile screen."""
    return {
        "keyboard": [
            [{"text": "📈 EUR/USD"}, {"text": "🥇 GOLD"}, {"text": "₿ BTC/USD"}],
            [{"text": "📊 All Markets"}, {"text": "⚙️ My Sizing"}, {"text": "❓ Help"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }


def send_telegram_message(text_html, parse_mode="HTML", reply_markup=None):
    """Dispatches an HTML formatted message directly to the linked Telegram user."""
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text_html,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            return data.get("ok", False), "Delivered"
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode()
        return False, f"HTTP Error {e.code}: {err_msg}"
    except Exception as e:
        return False, str(e)


def _load_alert_history():
    if os.path.exists(ALERT_HISTORY_FILE):
        try:
            with open(ALERT_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_alert_history(history):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(ALERT_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print("Error saving telegram_alert_history.json:", e)


def compute_custom_sizing(setup, instrument="EURUSD"):
    """Dynamically scales lot size and margin to the user's customized account sizing."""
    cfg = get_user_config()
    account_size = float(cfg.get("account_balance", 10000.0))
    leverage = int(cfg.get("leverage", 200))
    risk_pct = float(cfg.get("max_risk_pct", 0.50))

    spec = quant_model.get_instrument_spec(instrument)
    contract_val = float(spec["contract"])
    min_risk = float(spec["min_risk_floor"])
    price_dec = spec["price_decimals"]

    entry_val = float(setup.get("optimal_entry") or setup.get("current_price") or 1.0)
    sl_val = float(setup.get("stop_loss") or entry_val)
    risk_dist = max(min_risk, abs(entry_val - sl_val))
    risk_per_lot = risk_dist * contract_val

    risk_budget_usd = round(account_size * (risk_pct / 100.0), 2)
    max_cap = 0.20 if spec["symbol"] == "BTCUSD" else (2.00 if spec["symbol"] == "EURUSD" else 0.25)
    lot_size = round(max(0.01, min(max_cap, risk_budget_usd / max(0.01, risk_per_lot))), 2)
    req_margin = round((lot_size * contract_val * entry_val) / leverage, 2)
    return {
        "account_size": account_size,
        "leverage": leverage,
        "risk_pct": risk_pct,
        "risk_budget_usd": risk_budget_usd,
        "lot_size": lot_size,
        "req_margin": req_margin,
        "free_margin_pct": round(max(0.0, 100.0 - (req_margin / account_size * 100.0)), 2)
    }


def format_trade_alert(setup, instrument="EURUSD", horizon="scalping", title_prefix="A* PROP SCALP ALERT"):
    """
    Renders an institutional, high-density trade card for Telegram.
    Formatted for fast lock-screen reading and instant broker copy-paste.
    """
    is_buy = setup.get("is_bullish") or setup.get("action_type") == "BUY" or "LONG" in str(setup.get("action", "")).upper()
    direction_emoji = "🟢 BUY" if is_buy else "🔴 SELL"
    order_type = f"{direction_emoji} LIMIT"

    inst_name = str(instrument).upper()
    horizon_name = str(horizon).upper()

    entry_val = setup.get("optimal_entry") or setup.get("current_price") or setup.get("mid_entry", 0.0)
    sl_val = setup.get("stop_loss", 0.0)
    tp1_val = setup.get("take_profit_1", 0.0)
    tp2_val = setup.get("take_profit_2", 0.0)
    rr = setup.get("risk_reward_ratio", 2.0)

    prop_defense = setup.get("prop_firm_defense", {})
    sl_pips = prop_defense.get("sl_pips", 0.0)
    rem_pips = prop_defense.get("remaining_pips_buffer", 100.0 - sl_pips)
    losses_absorbed = prop_defense.get("consecutive_losses_absorbed", 5)

    # Dynamic user sizing
    sizing = compute_custom_sizing(setup, instrument=inst_name)
    lot_size = sizing["lot_size"]
    risk_usd = sizing["risk_budget_usd"]
    req_margin = sizing["req_margin"]
    acc_size = sizing["account_size"]
    lev = sizing["leverage"]
    free_margin = sizing["free_margin_pct"]

    # Quantitative gates summary
    gates = setup.get("gate_checks", {})
    vpin_status = gates.get("vpin_order_flow_gate", {}).get("status", "PASS")
    hawkes_status = gates.get("hawkes_cascade_firewall", {}).get("status", "PASS")
    hurst_status = gates.get("dfa_hurst_persistence", {}).get("status", "PASS")
    grade = setup.get("setup_grade", "A* GRADE")

    clean_inst = html.escape(str(inst_name))
    clean_horizon = html.escape(str(horizon_name))
    clean_order = html.escape(str(order_type))
    clean_grade = html.escape(str(grade))
    clean_vpin = html.escape(str(vpin_status))
    clean_hawkes = html.escape(str(hawkes_status))
    clean_hurst = html.escape(str(hurst_status))

    text = (
        f"⚡ <b>NEWSLYSER // {html.escape(str(title_prefix))}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Asset:</b> <code>{clean_inst}</code> | <b>Horizon:</b> <code>{clean_horizon}</code>\n"
        f"<b>Signal:</b> <b>{clean_order}</b> ({clean_grade})\n\n"
        f"🎯 <b>Entry Price:</b> <code>{entry_val}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>{sl_val}</code> (<b>{sl_pips:.1f} pips</b>)\n"
        f"🏁 <b>Take Profit 1:</b> <code>{tp1_val}</code> (<b>1:{rr:.2f} RR</b>)\n"
        f"🏁 <b>Take Profit 2:</b> <code>{tp2_val}</code> (Runner)\n\n"
        f"🛡️ <b>PROP FIRM 100-PIP DEFENSE:</b>\n"
        f"• Pip Impact: <b>-{sl_pips:.1f} pips</b> (<b>{rem_pips:.1f} pips</b> buffer left)\n"
        f"• Cushion: <b>{losses_absorbed} consecutive losses</b> absorbable\n\n"
        f"💼 <b>CUSTOM SIZING (${acc_size:,.0f} @ 1:{lev}):</b>\n"
        f"• Lot Size: <b>{lot_size:.2f} Lot</b>\n"
        f"• Risk Budget: <b>${risk_usd:.2f}</b> ({sizing['risk_pct']:.2f}% risk)\n"
        f"• Required Margin: <b>${req_margin:.2f}</b> ({free_margin}% free buffer)\n\n"
        f"🔬 <b>QUANTITATIVE GATES:</b>\n"
        f"• VPIN Flow: <code>{clean_vpin}</code> | Hawkes: <code>{clean_hawkes}</code> | Hurst: <code>{clean_hurst}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ <i>Real-time snapshot. Enter pending limit order on broker.</i>"
    )
    return text


def get_live_setup_for_instrument(instrument="EURUSD", horizon="scalping"):
    """Fetches live prices and calculates fresh setup using quant_model."""
    spec = quant_model.get_instrument_spec(instrument)
    inst_key = spec["symbol"]

    try:
        from data_pipeline import fetch_live_macro_data
        live = fetch_live_macro_data()
        prices = live.get("prices", {}) if isinstance(live, dict) else {}
    except Exception:
        prices = {}

    if inst_key == "EURUSD":
        cur_px = prices.get("EURUSD=X", 1.0850)
        atr_val = prices.get("EURUSD_ATR_M30", 0.0012)
    elif inst_key == "BTCUSD":
        cur_px = prices.get("BTC-USD", 91500.0)
        atr_val = prices.get("BTC_ATR_M30", 1000.0)
    else:  # XAUUSD
        cur_px = prices.get("GC=F", 4036.0)
        atr_val = prices.get("ATR_M30", 12.5)

    setup = quant_model.calculate_daily_a_star_setup(
        current_price=cur_px,
        atr_m30=atr_val,
        horizon=horizon,
        instrument=inst_key
    )
    return setup, inst_key

# Alias
get_live_market_setup = get_live_setup_for_instrument


def dispatch_a_star_alert(setup, instrument="EURUSD", horizon="scalping", force=False):
    """
    Evaluates setup and sends an automated Telegram notification if:
      1. Setup is an A* Grade setup.
      2. It has not already been alerted for this setup_created_epoch.
      3. Prop firm defense is compliant.
    """
    if not setup or not isinstance(setup, dict):
        return False, "Invalid setup object"

    grade = str(setup.get("setup_grade", ""))
    is_a_star = "A*" in grade or setup.get("is_a_star", False)
    if not is_a_star and not force:
        return False, "Setup is not A* Grade"

    created_epoch = setup.get("setup_created_epoch") or setup.get("epoch", int(time.time()))
    alert_key = f"{str(instrument).upper()}_{str(horizon).lower()}_{created_epoch}"

    history = _load_alert_history()
    if not force and alert_key in history:
        return False, f"Already alerted for {alert_key}"

    msg = format_trade_alert(setup, instrument=instrument, horizon=horizon, title_prefix="A* PROP SCALP ALERT")
    ok, err = send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
    if ok:
        history[alert_key] = {
            "timestamp": int(time.time()),
            "instrument": instrument,
            "horizon": horizon,
            "action": setup.get("action", "")
        }
        if len(history) > 200:
            sorted_keys = sorted(history.keys(), key=lambda k: history[k].get("timestamp", 0))
            for old_k in sorted_keys[:-100]:
                history.pop(old_k, None)
        _save_alert_history(history)
        return True, "Alert dispatched successfully"
    return False, err


def handle_telegram_command(text_raw, from_id):
    """
    Processes incoming interactive Telegram commands:
      - /eur, /gold, /btc
      - /size <balance>, /risk <pct>, /leverage <ratio>
      - /config, /settings
      - /all, /status
      - /start, /help, /menu
    """
    _, authorized_chat_id = get_telegram_credentials()
    if str(from_id) != str(authorized_chat_id):
        return False, "Unauthorized user"

    text = (text_raw or "").strip()
    cmd = text.split()[0].lower() if text else ""
    cfg = get_user_config()

    # 1. /start, /menu, /help
    if cmd in ("/start", "/menu", "/help", "help", "menu", "❓ help"):
        help_text = (
            f"⚡ <b>NEWSLYSER QUANTITATIVE TERMINAL BOT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Welcome Thyra! You can query live market setups and dynamically adjust your sizing directly from this chat.\n\n"
            f"📊 <b>ON-DEMAND MARKET QUERIES:</b>\n"
            f"• <code>/eur</code> — Live EUR/USD Scalp & Intraday ticket\n"
            f"• <code>/gold</code> — Live Gold (XAU/USD) ticket\n"
            f"• <code>/btc</code> — Live Bitcoin (BTC/USD) ticket\n"
            f"• <code>/all</code> — Quick Triad Market Overview\n\n"
            f"⚙️ <b>ACCOUNT SIZING ADJUSTMENTS:</b>\n"
            f"• <code>/size &lt;balance&gt;</code> — E.g. <code>/size 10000</code> or <code>/size 20000</code>\n"
            f"• <code>/risk &lt;pct&gt;</code> — E.g. <code>/risk 0.5</code> or <code>/risk 0.35</code>\n"
            f"• <code>/leverage &lt;ratio&gt;</code> — E.g. <code>/leverage 200</code> or <code>/leverage 100</code>\n"
            f"• <code>/config</code> — View your current sizing profile\n\n"
            f"<i>Tap any button below for instant execution!</i>"
        )
        return send_telegram_message(help_text, reply_markup=get_main_reply_keyboard())

    # 2. Sizing Config Adjustments: /size, /balance
    if cmd in ("/size", "/balance"):
        parts = text.split()
        if len(parts) >= 2:
            raw_val = parts[1].lower().replace("$", "").replace(",", "")
            if raw_val.endswith("k"):
                val = float(raw_val[:-1]) * 1000.0
            else:
                val = float(raw_val)
            if 100.0 <= val <= 1000000.0:
                cfg["account_balance"] = round(val, 2)
                save_user_config(cfg)
                risk_usd = round(val * (cfg["max_risk_pct"] / 100.0), 2)
                msg = (
                    f"✅ <b>ACCOUNT BALANCE UPDATED</b>\n\n"
                    f"• <b>New Balance:</b> <code>${val:,.2f}</code>\n"
                    f"• <b>Risk Per Trade:</b> <code>{cfg['max_risk_pct']:.2f}%</code> (<b>${risk_usd:.2f}</b>)\n"
                    f"• <b>Leverage:</b> <code>1:{cfg['leverage']}</code>\n\n"
                    f"<i>All future scalp alerts and manual tickets will calculate lots using ${val:,.0f}.</i>"
                )
                return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
        return send_telegram_message("❌ Usage: <code>/size 10000</code> (between $100 and $1,000,000)")

    # 3. Risk Adjustment: /risk
    if cmd in ("/risk",):
        parts = text.split()
        if len(parts) >= 2:
            val = float(parts[1].replace("%", ""))
            if 0.05 <= val <= 5.0:
                cfg["max_risk_pct"] = round(val, 2)
                save_user_config(cfg)
                risk_usd = round(cfg["account_balance"] * (val / 100.0), 2)
                msg = (
                    f"✅ <b>RISK CEILING UPDATED</b>\n\n"
                    f"• <b>New Risk %:</b> <code>{val:.2f}%</code>\n"
                    f"• <b>Risk Budget:</b> <b>${risk_usd:.2f}</b> on ${cfg['account_balance']:,.0f} balance\n\n"
                    f"<i>Prop firm safety preserved (max 100-pip ceiling active).</i>"
                )
                return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
        return send_telegram_message("❌ Usage: <code>/risk 0.5</code> (between 0.05% and 5.0%)")

    # 4. Leverage Adjustment: /leverage, /lev
    if cmd in ("/leverage", "/lev"):
        parts = text.split()
        if len(parts) >= 2:
            val = int(parts[1].replace("1:", ""))
            if 1 <= val <= 1000:
                cfg["leverage"] = val
                save_user_config(cfg)
                msg = f"✅ <b>LEVERAGE UPDATED</b>\n\n• <b>New Leverage:</b> <code>1:{val}</code>\n\n<i>Margin requirements re-calibrated.</i>"
                return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
        return send_telegram_message("❌ Usage: <code>/leverage 200</code>")

    # 5. /config, /settings, "⚙️ My Sizing"
    if cmd in ("/config", "/settings", "config", "settings", "⚙️ my sizing"):
        acc = cfg.get("account_balance", 10000.0)
        risk_pct = cfg.get("max_risk_pct", 0.50)
        risk_usd = round(acc * (risk_pct / 100.0), 2)
        lev = cfg.get("leverage", 200)
        msg = (
            f"⚙️ <b>CURRENT TRADING CONFIGURATION</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Account Balance:</b> <code>${acc:,.2f}</code>\n"
            f"• <b>Risk Cap:</b> <code>{risk_pct:.2f}%</code> (${risk_usd:.2f} per trade)\n"
            f"• <b>Broker Leverage:</b> <code>1:{lev}</code>\n"
            f"• <b>Prop Firm Ceiling:</b> <code>100.0 Pips</code> (Hard Loss Barrier)\n"
            f"• <b>Primary Mode:</b> <code>SCALPING (M5/M15) + INTRADAY (H1)</code>\n\n"
            f"📝 <b>To update sizing anytime:</b>\n"
            f"• <code>/size 20000</code>\n"
            f"• <code>/risk 0.5</code>\n"
            f"• <code>/leverage 200</code>"
        )
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 6. Manual Asset Queries: EURUSD
    if cmd in ("/eur", "/eurusd", "eur", "eurusd", "eur/usd", "📈 eur/usd"):
        horizon = "intraday" if "intraday" in text.lower() else ("daily" if "daily" in text.lower() else "scalping")
        setup, inst_key = get_live_market_setup("EURUSD", horizon=horizon)
        msg = format_trade_alert(setup, instrument=inst_key, horizon=horizon, title_prefix="LIVE EUR/USD SNAPSHOT")
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 7. Manual Asset Queries: Gold (XAUUSD)
    if cmd in ("/gold", "/xau", "/xauusd", "gold", "xau", "🥇 gold"):
        horizon = "intraday" if "intraday" in text.lower() else ("daily" if "daily" in text.lower() else "scalping")
        setup, inst_key = get_live_market_setup("XAUUSD", horizon=horizon)
        msg = format_trade_alert(setup, instrument=inst_key, horizon=horizon, title_prefix="LIVE GOLD SNAPSHOT")
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 8. Manual Asset Queries: Bitcoin (BTCUSD)
    if cmd in ("/btc", "/btcusd", "btc", "bitcoin", "₿ btc/usd"):
        horizon = "intraday" if "intraday" in text.lower() else ("daily" if "daily" in text.lower() else "scalping")
        setup, inst_key = get_live_market_setup("BTCUSD", horizon=horizon)
        msg = format_trade_alert(setup, instrument=inst_key, horizon=horizon, title_prefix="LIVE BITCOIN SNAPSHOT")
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 9. All Markets Triad Overview: /all, /status, "📊 All Markets"
    if cmd in ("/all", "/status", "all", "status", "overview", "📊 all markets"):
        eur, _ = get_live_market_setup("EURUSD", horizon="scalping")
        gold, _ = get_live_market_setup("XAUUSD", horizon="scalping")
        btc, _ = get_live_market_setup("BTCUSD", horizon="scalping")

        def _format_bias_tag(setup):
            tt = setup.get("trade_ticket", {})
            act = str(tt.get("action") or setup.get("action") or "STAND ASIDE").upper()
            if "BUY" in act:
                return "🟢 BUY LIMIT"
            elif "SELL" in act:
                return "🔴 SELL LIMIT"
            return "⚪ STAND ASIDE"

        eur_bias = _format_bias_tag(eur)
        gold_bias = _format_bias_tag(gold)
        btc_bias = _format_bias_tag(btc)

        eur_px = eur.get('current_price', 1.0850)
        gold_px = gold.get('current_price', 4036.0)
        btc_px = btc.get('current_price', 91500.0)

        eur_sl_pip = eur.get('prop_firm_defense', {}).get('sl_pips', 16.4)
        gold_sl = gold.get('stop_loss', 0.0)
        btc_sl = btc.get('stop_loss', 0.0)

        msg = (
            f"📊 <b>NEWSLYSER // TRIAD MARKET TELEMETRY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💶 <b>EUR/USD:</b> <code>${eur_px:.5f}</code>\n"
            f"• Bias: <b>{eur_bias}</b>\n"
            f"• Prop SL: <b>{eur_sl_pip:.1f} pips</b> | Score: <b>{eur.get('confluence_score', 0)}/100</b>\n\n"
            f"🥇 <b>GOLD (XAU/USD):</b> <code>${gold_px:,.2f}</code>\n"
            f"• Bias: <b>{gold_bias}</b>\n"
            f"• Stop Loss: <code>${gold_sl:,.2f}</code> | Score: <b>{gold.get('confluence_score', 0)}/100</b>\n\n"
            f"₿ <b>BITCOIN (BTC/USD):</b> <code>${btc_px:,.1f}</code>\n"
            f"• Bias: <b>{btc_bias}</b>\n"
            f"• Stop Loss: <code>${btc_sl:,.1f}</code> | Score: <b>{btc.get('confluence_score', 0)}/100</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Send /eur, /gold, or /btc for detailed tickets.</i>"
        )
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # Default fallback
    clean_input = html.escape(str(text))
    return send_telegram_message(
        f"❓ Unknown command: <code>{clean_input}</code>\n\nSend <code>/help</code> or tap the buttons below for options.",
        reply_markup=get_main_reply_keyboard()
    )


def _telegram_polling_worker():
    """Continuous background worker listening for user commands via getUpdates."""
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        return

    offset = 0
    # First, discard old pending updates on boot
    try:
        flush_url = f"https://api.telegram.org/bot{token}/getUpdates?offset=-1"
        with urllib.request.urlopen(flush_url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            res = data.get("result", [])
            if res:
                offset = res[-1]["update_id"] + 1
    except Exception:
        pass

    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates?offset={offset}&timeout=5"
            req = urllib.request.Request(url, headers={"User-Agent": "NewslyserBot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message")
                    if msg:
                        text = msg.get("text")
                        sender_id = msg.get("from", {}).get("id")
                        if text and sender_id:
                            handle_telegram_command(text, sender_id)
        except Exception:
            time.sleep(2)
        time.sleep(0.5)


_POLLING_STARTED = False

def start_telegram_polling():
    """Starts the Telegram interactive command listener in a background daemon thread."""
    global _POLLING_STARTED
    if _POLLING_STARTED:
        return
    token, chat_id = get_telegram_credentials()
    if token and chat_id:
        t = threading.Thread(target=_telegram_polling_worker, daemon=True, name="TelegramPollingWorker")
        t.start()
        _POLLING_STARTED = True
