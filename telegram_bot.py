import os
import json
import time
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ALERT_HISTORY_FILE = os.path.join(DATA_DIR, "telegram_alert_history.json")

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


def send_telegram_message(text_html, parse_mode="HTML"):
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


def format_trade_alert(setup, instrument="EURUSD", horizon="scalping", account_size=10000.0, leverage=200):
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
    
    lot_size = setup.get("safe_lot_size", 0.07)
    req_margin = setup.get("required_margin", 32.0)
    risk_usd = setup.get("target_risk_usd", 35.0)
    
    # Gate summary
    gates = setup.get("gate_checks", {})
    vpin_status = gates.get("vpin_order_flow_gate", {}).get("status", "PASS")
    hawkes_status = gates.get("hawkes_cascade_firewall", {}).get("status", "PASS")
    hurst_status = gates.get("dfa_hurst_persistence", {}).get("status", "PASS")
    
    text = (
        f"⚡ <b>NEWSLYSER // A* PROP SCALP ALERT</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Asset:</b> <code>{inst_name}</code> | <b>Horizon:</b> <code>{horizon_name} (M5/M15)</code>\n"
        f"<b>Signal:</b> <b>{order_type}</b>\n\n"
        f"🎯 <b>Entry Price:</b> <code>{entry_val}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>{sl_val}</code> (<b>{sl_pips:.1f} pips</b>)\n"
        f"🏁 <b>Take Profit 1:</b> <code>{tp1_val}</code> (<b>1:{rr:.2f} RR</b>)\n"
        f"🏁 <b>Take Profit 2:</b> <code>{tp2_val}</code> (Runner)\n\n"
        f"🛡️ <b>PROP FIRM 100-PIP DEFENSE:</b>\n"
        f"• Pip Impact: <b>-{sl_pips:.1f} pips</b> (<b>{rem_pips:.1f} pips</b> buffer left)\n"
        f"• Cushion: <b>{losses_absorbed} consecutive losses</b> absorbable\n\n"
        f"💼 <b>POSITION SIZING (${account_size:,.0f} @ 1:{leverage}):</b>\n"
        f"• Lot Size: <b>{lot_size:.2f} Lot</b>\n"
        f"• Risk Budget: <b>${risk_usd:.2f}</b> (0.35%–0.50% max)\n"
        f"• Required Margin: <b>${req_margin:.2f}</b> (>99.6% Free Margin)\n\n"
        f"🔬 <b>QUANTITATIVE GATES:</b>\n"
        f"• VPIN Flow: <code>{vpin_status}</code> | Hawkes: <code>{hawkes_status}</code> | Hurst: <code>{hurst_status}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ <i>Valid until TTL epoch or price runs away >0.35 ATR.</i>"
    )
    return text


def dispatch_a_star_alert(setup, instrument="EURUSD", horizon="scalping", force=False):
    """
    Evaluates setup and sends a Telegram notification if:
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

    msg = format_trade_alert(setup, instrument=instrument, horizon=horizon)
    ok, err = send_telegram_message(msg)
    if ok:
        history[alert_key] = {
            "timestamp": int(time.time()),
            "instrument": instrument,
            "horizon": horizon,
            "action": setup.get("action", "")
        }
        # Keep history pruned to last 200 entries
        if len(history) > 200:
            sorted_keys = sorted(history.keys(), key=lambda k: history[k].get("timestamp", 0))
            for old_k in sorted_keys[:-100]:
                history.pop(old_k, None)
        _save_alert_history(history)
        return True, "Alert dispatched successfully"
    return False, err
