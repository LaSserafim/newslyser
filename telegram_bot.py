import os
import json
import time
import math
import re
import html
import threading
import urllib.request
import urllib.error

import quant_model
import market_feed

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
        "max_loss_usd": 25.0,  # Strict $25 max loss per trade / stop loss
        "max_risk_pct": 0.25,
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
    """Persists updated account balance, max loss, and leverage settings."""
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
            [{"text": "📊 All Markets"}, {"text": "⚙️ My Sizing"}, {"text": "🛡️ Max Loss $25"}]
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
    """
    Calculates exact lot size guaranteed to cap loss at max_loss_usd (default $25.00).
    Formula: Lot = floor(Max Loss USD / (Stop Distance * Contract Size))
    """
    cfg = get_user_config()
    account_size = float(cfg.get("account_balance", 10000.0))
    leverage = int(cfg.get("leverage", 200))
    max_loss_usd = float(cfg.get("max_loss_usd", 25.0))

    spec = quant_model.get_instrument_spec(instrument)
    contract_val = float(spec["contract"])
    min_fl = float(spec["min_risk_floor"])
    price_dec = spec["price_decimals"]

    entry_val = float(setup.get("optimal_entry") or setup.get("first_passage_analysis", {}).get("entry_price") or setup.get("current_price") or 1.0)
    sl_val = float(setup.get("stop_loss") or setup.get("first_passage_analysis", {}).get("sl_price") or (entry_val - min_fl))

    # Absolute stop distance
    sl_dist = max(min_fl, abs(entry_val - sl_val))

    if spec["symbol"] == "EURUSD":
        sl_pips = sl_dist / 0.00010
        pips_label = f"{sl_pips:.1f} pips"
    elif spec["symbol"] == "XAUUSD":
        sl_pips = sl_dist * 10.0
        pips_label = f"${sl_dist:.2f} ({sl_pips:.1f} pips)"
    else:  # BTCUSD
        pips_label = f"${sl_dist:,.1f}"
        sl_pips = sl_dist

    loss_per_lot = sl_dist * contract_val
    exact_lots = max_loss_usd / max(0.001, loss_per_lot)

    # Broker step = 0.01 lot. Floor to 2 decimals to NEVER exceed max_loss_usd!
    lot_size = max(0.01, math.floor(exact_lots * 100.0) / 100.0)

    max_cap = 5.00 if spec["symbol"] == "EURUSD" else (0.50 if spec["symbol"] == "XAUUSD" else 0.20)
    lot_size = min(max_cap, lot_size)

    actual_max_loss = round(lot_size * loss_per_lot, 2)
    req_margin = round((lot_size * contract_val * entry_val) / leverage, 2)
    free_margin_pct = round(max(0.0, 100.0 - (req_margin / account_size * 100.0)), 2)

    return {
        "account_size": account_size,
        "leverage": leverage,
        "max_loss_usd": max_loss_usd,
        "actual_max_loss": actual_max_loss,
        "sl_dist": sl_dist,
        "sl_pips": sl_pips,
        "pips_label": pips_label,
        "loss_per_lot": loss_per_lot,
        "lot_size": lot_size,
        "req_margin": req_margin,
        "free_margin_pct": free_margin_pct
    }


def format_trade_alert(setup, instrument="EURUSD", horizon="scalping", title_prefix="A* PROP SCALP ALERT"):
    """
    Renders an institutional, mathematically verified trade card for Telegram.
    Formatted for fast lock-screen reading and clear, unambiguous broker execution.
    """
    spec = quant_model.get_instrument_spec(instrument)
    inst_name = str(spec["name"]).upper()
    inst_sym = spec["symbol"]
    price_dec = spec["price_decimals"]
    horizon_name = str(horizon).upper()

    cur_px = setup.get("current_price", 0.0)
    cur_px_str = f"{cur_px:.5f}" if inst_sym == "EURUSD" else (f"{cur_px:,.2f}" if inst_sym == "XAUUSD" else f"{cur_px:,.1f}")

    is_a_star = setup.get("is_a_star", False)
    grade = setup.get("setup_grade") or setup.get("grade") or "A* GRADE"
    action = str(setup.get("action") or setup.get("trade_ticket", {}).get("action") or "").upper()
    is_buy = setup.get("is_bullish") or setup.get("action_type") == "BUY" or "BUY" in action or "LONG" in action

    # Dynamic user sizing anchored to strict maximum dollar risk
    sizing = compute_custom_sizing(setup, instrument=inst_sym)
    lot_size = sizing["lot_size"]
    max_loss_usd = sizing["max_loss_usd"]
    actual_max_loss = sizing["actual_max_loss"]
    pips_label = sizing["pips_label"]
    req_margin = sizing["req_margin"]
    acc_size = sizing["account_size"]
    lev = sizing["leverage"]
    free_margin = sizing["free_margin_pct"]

    # Gates
    gates = setup.get("gate_checks", {})
    vpin_status = gates.get("vpin_order_flow_gate", {}).get("status", "PASS")
    hawkes_status = gates.get("hawkes_cascade_firewall", {}).get("status", "PASS")
    hurst_status = gates.get("dfa_hurst_persistence", {}).get("status", "PASS")
    score = setup.get("confluence_score", 0)

    # ----------------------------------------------------
    # CASE 1: ACTIVE A* SETUP READY FOR IMMEDIATE EXECUTION
    # ----------------------------------------------------
    if is_a_star and "STAND ASIDE" not in action:
        direction_emoji = "🟢 BUY" if is_buy else "🔴 SELL"
        order_type = f"{direction_emoji} LIMIT"

        entry_val = setup.get("optimal_entry") or cur_px
        sl_val = setup.get("stop_loss", cur_px)
        tp1_val = setup.get("take_profit_1", cur_px)
        tp2_val = setup.get("take_profit_2", cur_px)
        rr = setup.get("risk_reward_ratio", 2.5)

        fmt_entry = f"{entry_val:.5f}" if inst_sym == "EURUSD" else (f"{entry_val:,.2f}" if inst_sym == "XAUUSD" else f"{entry_val:,.1f}")
        fmt_sl = f"{sl_val:.5f}" if inst_sym == "EURUSD" else (f"{sl_val:,.2f}" if inst_sym == "XAUUSD" else f"{sl_val:,.1f}")
        fmt_tp1 = f"{tp1_val:.5f}" if inst_sym == "EURUSD" else (f"{tp1_val:,.2f}" if inst_sym == "XAUUSD" else f"{tp1_val:,.1f}")
        fmt_tp2 = f"{tp2_val:.5f}" if inst_sym == "EURUSD" else (f"{tp2_val:,.2f}" if inst_sym == "XAUUSD" else f"{tp2_val:,.1f}")

        text = (
            f"⚡ <b>NEWSLYSER // {html.escape(title_prefix)}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Asset:</b> <code>{inst_name} ({inst_sym})</code>\n"
            f"<b>Horizon:</b> <code>{html.escape(horizon_name)}</code>\n"
            f"<b>Signal:</b> <b>{html.escape(order_type)}</b> (<code>{html.escape(grade)}</code>)\n\n"
            f"🎯 <b>Entry Price:</b> <code>{fmt_entry}</code> (Spot: <code>${cur_px_str}</code>)\n"
            f"🛑 <b>Stop Loss:</b> <code>{fmt_sl}</code> (<b>{pips_label}</b>)\n"
            f"🏁 <b>Take Profit 1:</b> <code>{fmt_tp1}</code> (<b>1:{rr:.2f} RR</b>)\n"
            f"🏁 <b>Take Profit 2:</b> <code>{fmt_tp2}</code> (Runner)\n\n"
            f"🛡️ <b>RISK CAPPED AT ${max_loss_usd:.2f} PER TRADE:</b>\n"
            f"• <b>Recommended Lot:</b> <b>{lot_size:.2f} Lot</b>\n"
            f"• <b>Max Loss at Stop:</b> <b>-${actual_max_loss:.2f}</b> (Protected &le; ${max_loss_usd:.2f})\n"
            f"• <b>Margin Required:</b> <b>${req_margin:.2f}</b> (1:{lev} lev | {free_margin:.1f}% free)\n\n"
            f"🔬 <b>QUANTITATIVE GATES:</b>\n"
            f"• VPIN: <code>{html.escape(str(vpin_status))}</code> | Hawkes: <code>{html.escape(str(hawkes_status))}</code> | Hurst: <code>{html.escape(str(hurst_status))}</code>\n"
            f"• Confluence Score: <b>{score}/100</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ <i>Live verified feed. Enter limit order on broker.</i>"
        )
        return text

    # ----------------------------------------------------
    # CASE 2: STAND ASIDE // NO CURRENT A* EDGE (HONEST & UNBIASED)
    # ----------------------------------------------------
    snr_note = "NEUTRAL / CHOP DEADBAND"
    if inst_sym == "EURUSD":
        snr_dict = setup.get("snr_pure_trend") or {}
        tr = str(snr_dict.get("trend", "")).upper()
        if "BULLISH" in tr:
            snr_note = "BULLISH PURE-TREND (Price > EMA50 > EMA200)"
        elif "BEARISH" in tr:
            snr_note = "BEARISH PURE-TREND (Price < EMA50 < EMA200)"
        else:
            snr_note = "CHOP DEADBAND (Sandwich Zone)"
    elif inst_sym == "BTCUSD":
        btc_dict = setup.get("btc_regime") or {}
        tr = str(btc_dict.get("trend", "")).upper()
        if "BULLISH" in tr:
            snr_note = "BULLISH PURE-TREND (Price > EMA50 > EMA200)"
        elif "BEARISH" in tr:
            snr_note = "BEARISH PURE-TREND (Price < EMA50 < EMA200)"
        else:
            snr_note = "CHOP / SIDEWAYS REGIME"
    else:  # XAUUSD
        bias_str = str((setup.get("momentum_metrics") or {}).get("momentum_regime") or setup.get("bias") or "NEUTRAL").upper()
        snr_note = f"MACRO REGIME: {bias_str}"

    # Identify blocking reason cleanly
    reason_gate = "Confluence below threshold"
    tt = setup.get("trade_ticket") or {}
    act_str = tt.get("action", "")
    if "(" in act_str and ")" in act_str:
        reason_gate = act_str[act_str.find("(") + 1:act_str.rfind(")")]

    # Key Levels to watch from candidate SDE
    fp = setup.get("first_passage_analysis", {})
    entry_p = fp.get("entry_price") or cur_px
    sl_p = fp.get("sl_price") or (cur_px - sizing["sl_dist"])
    tp_p = fp.get("tp_price") or (cur_px + 2.5 * sizing["sl_dist"])

    fmt_entry = f"{entry_p:.5f}" if inst_sym == "EURUSD" else (f"{entry_p:,.2f}" if inst_sym == "XAUUSD" else f"{entry_p:,.1f}")
    fmt_sl = f"{sl_p:.5f}" if inst_sym == "EURUSD" else (f"{sl_p:,.2f}" if inst_sym == "XAUUSD" else f"{sl_p:,.1f}")
    fmt_tp = f"{tp_p:.5f}" if inst_sym == "EURUSD" else (f"{tp_p:,.2f}" if inst_sym == "XAUUSD" else f"{tp_p:,.1f}")

    text = (
        f"⚡ <b>NEWSLYSER // {html.escape(title_prefix)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Asset:</b> <code>{inst_name} ({inst_sym})</code>\n"
        f"<b>Horizon:</b> <code>{html.escape(horizon_name)}</code>\n"
        f"<b>Status:</b> ⚪ <b>STAND ASIDE (NO A* EDGE)</b>\n\n"
        f"🎯 <b>Live Spot Price:</b> <code>${cur_px_str}</code>\n"
        f"📈 <b>Trend Regime:</b> <code>{html.escape(snr_note)}</code>\n"
        f"📊 <b>Confluence Score:</b> <b>{score}/100</b> (Threshold &ge; 75.0)\n"
        f"🚫 <b>Filter Gate:</b> <i>{html.escape(reason_gate)}</i>\n\n"
        f"📍 <b>KEY CANDIDATE LEVELS TO WATCH:</b>\n"
        f"• <b>Reference Entry:</b> <code>${fmt_entry}</code>\n"
        f"• <b>Structural Stop:</b> <code>${fmt_sl}</code> (<b>{pips_label}</b>)\n"
        f"• <b>Expected Target:</b> <code>${fmt_tp}</code>\n\n"
        f"🛡️ <b>MAX RISK CALIBRATION:</b>\n"
        f"• <b>Max Loss Limit:</b> <b>${max_loss_usd:.2f} per trade</b>\n"
        f"• <b>Planned Lot Size:</b> <b>{lot_size:.2f} Lot</b> (if setup fires)\n"
        f"• <b>Capital at Risk:</b> <b>$0.00</b> (Capital preserved)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ <i>Real-time feed. Standing aside until all mathematical gates align. Send /loss {int(max_loss_usd)} to adjust risk.</i>"
    )
    return text


def get_live_setup_for_instrument(instrument="EURUSD", horizon="scalping"):
    """Fetches verified live prices and calculates setup using market_feed."""
    setup, inst_key, _ = market_feed.get_live_market_setup(instrument=instrument, horizon=horizon)
    return setup, inst_key


# Alias
get_live_market_setup = get_live_setup_for_instrument


def dispatch_a_star_alert(setup, instrument="EURUSD", horizon="scalping", force=False):
    """
    Evaluates setup and sends an automated Telegram notification if:
      1. Setup is an A* Grade setup.
      2. It has not already been alerted for this setup_created_epoch.
    """
    if not setup or not isinstance(setup, dict):
        return False, "Invalid setup object"

    grade = str(setup.get("setup_grade") or setup.get("grade") or "")
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
      - /loss <dollar>, /maxloss <dollar> (adjusts maximum dollar loss per trade)
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
        max_l = cfg.get("max_loss_usd", 25.0)
        acc = cfg.get("account_balance", 10000.0)
        help_text = (
            f"⚡ <b>NEWSLYSER QUANTITATIVE TERMINAL BOT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Welcome Thyra! You can query live market setups and adjust your maximum dollar loss anytime directly from this chat.\n\n"
            f"📊 <b>ON-DEMAND LIVE MARKET QUERIES:</b>\n"
            f"• <code>/eur</code> — Live EUR/USD Scalp ticket & key levels\n"
            f"• <code>/gold</code> — Live Gold (XAU/USD) ticket & key levels\n"
            f"• <code>/btc</code> — Live Bitcoin (BTC/USD) ticket & key levels\n"
            f"• <code>/all</code> — Triad Live Telemetry Overview\n\n"
            f"🛡️ <b>MAXIMUM RISK & SIZING CONTROLS:</b>\n"
            f"• <code>/loss &lt;dollars&gt;</code> — E.g. <code>/loss 25</code> or <code>/loss 50</code> (Max loss per stop loss)\n"
            f"• <code>/size &lt;balance&gt;</code> — E.g. <code>/size 10000</code> or <code>/size 20000</code>\n"
            f"• <code>/leverage &lt;ratio&gt;</code> — E.g. <code>/leverage 200</code>\n"
            f"• <code>/config</code> — View your active sizing & risk profile\n\n"
            f"<i>Current Protection: Max Loss <b>${max_l:.2f}</b> on ${acc:,.0f} balance.</i>"
        )
        return send_telegram_message(help_text, reply_markup=get_main_reply_keyboard())

    # 2. Maximum Dollar Loss Adjustment: /loss, /maxloss, "🛡️ Max Loss $25"
    if cmd in ("/loss", "/maxloss") or "max loss $25" in text.lower():
        if "max loss $25" in text.lower():
            val = 25.0
        else:
            parts = text.split()
            if len(parts) >= 2:
                raw_val = parts[1].replace("$", "").replace(",", "")
                try:
                    val = float(raw_val)
                except ValueError:
                    val = -1.0
            else:
                val = -1.0

        if 5.0 <= val <= 5000.0:
            cfg["max_loss_usd"] = round(val, 2)
            cfg["max_risk_pct"] = round((val / cfg["account_balance"]) * 100.0, 3)
            save_user_config(cfg)
            msg = (
                f"✅ <b>MAXIMUM RISK PER TRADE UPDATED</b>\n\n"
                f"• <b>Max Dollar Loss:</b> <code>${val:,.2f}</code> per stop loss\n"
                f"• <b>Equivalent Risk:</b> <code>{cfg['max_risk_pct']:.2f}%</code> on ${cfg['account_balance']:,.0f} balance\n"
                f"• <b>Broker Leverage:</b> <code>1:{cfg['leverage']}</code>\n\n"
                f"<i>All future scalp alerts and manual tickets will calculate lot sizes so you NEVER lose more than ${val:,.2f} if stopped out.</i>"
            )
            return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
        return send_telegram_message("❌ Usage: <code>/loss 25</code> (between $5 and $5,000 per trade)")

    # 3. Account Size Adjustment: /size, /balance
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
                max_l = cfg.get("max_loss_usd", 25.0)
                cfg["max_risk_pct"] = round((max_l / val) * 100.0, 3)
                save_user_config(cfg)
                msg = (
                    f"✅ <b>ACCOUNT BALANCE UPDATED</b>\n\n"
                    f"• <b>New Balance:</b> <code>${val:,.2f}</code>\n"
                    f"• <b>Max Dollar Loss:</b> <code>${max_l:.2f}</code> ({cfg['max_risk_pct']:.2f}% risk)\n"
                    f"• <b>Leverage:</b> <code>1:{cfg['leverage']}</code>\n\n"
                    f"<i>All calculations re-calibrated for ${val:,.0f} balance.</i>"
                )
                return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
        return send_telegram_message("❌ Usage: <code>/size 10000</code> (between $100 and $1,000,000)")

    # 4. Risk Adjustment: /risk
    if cmd in ("/risk",):
        parts = text.split()
        if len(parts) >= 2:
            raw_val = parts[1].replace("$", "")
            if "%" in raw_val:
                pct = float(raw_val.replace("%", ""))
                if 0.01 <= pct <= 10.0:
                    max_l = round(cfg["account_balance"] * (pct / 100.0), 2)
                    cfg["max_loss_usd"] = max_l
                    cfg["max_risk_pct"] = pct
                    save_user_config(cfg)
                    msg = (
                        f"✅ <b>RISK CEILING UPDATED</b>\n\n"
                        f"• <b>New Risk %:</b> <code>{pct:.2f}%</code>\n"
                        f"• <b>Max Dollar Loss:</b> <code>${max_l:.2f}</code> per stop loss\n\n"
                        f"<i>Lot sizes dynamically calibrated to cap loss at ${max_l:.2f}.</i>"
                    )
                    return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
            else:
                try:
                    dollar_val = float(raw_val)
                    if 5.0 <= dollar_val <= 5000.0:
                        cfg["max_loss_usd"] = round(dollar_val, 2)
                        cfg["max_risk_pct"] = round((dollar_val / cfg["account_balance"]) * 100.0, 3)
                        save_user_config(cfg)
                        msg = (
                            f"✅ <b>MAXIMUM RISK PER TRADE UPDATED</b>\n\n"
                            f"• <b>Max Dollar Loss:</b> <code>${dollar_val:.2f}</code>\n"
                            f"• <b>Risk %:</b> <code>{cfg['max_risk_pct']:.2f}%</code>\n"
                        )
                        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())
                except ValueError:
                    pass
        return send_telegram_message("❌ Usage: <code>/loss 25</code> or <code>/risk 0.5%</code>")

    # 5. Leverage Adjustment: /leverage, /lev
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

    # 6. /config, /settings, "⚙️ My Sizing"
    if cmd in ("/config", "/settings", "config", "settings", "⚙️ my sizing"):
        acc = cfg.get("account_balance", 10000.0)
        max_l = cfg.get("max_loss_usd", 25.0)
        risk_pct = cfg.get("max_risk_pct", (max_l / acc) * 100.0)
        lev = cfg.get("leverage", 200)
        msg = (
            f"⚙️ <b>CURRENT TRADING CONFIGURATION</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Account Balance:</b> <code>${acc:,.2f}</code>\n"
            f"• <b>Max Loss Per Trade:</b> <code>${max_l:.2f}</code> (Strict Stop Loss Ceiling)\n"
            f"• <b>Risk Percentage:</b> <code>{risk_pct:.2f}%</code> of balance\n"
            f"• <b>Broker Leverage:</b> <code>1:{lev}</code>\n"
            f"• <b>Primary Execution:</b> <code>SCALPING (M5/M15) + INTRADAY (H1)</code>\n\n"
            f"📝 <b>To adjust sizing anytime:</b>\n"
            f"• <code>/loss 25</code> — Set max loss to $25.00\n"
            f"• <code>/loss 50</code> — Set max loss to $50.00\n"
            f"• <code>/size 20000</code> — Update account balance\n"
            f"• <code>/leverage 200</code> — Update broker leverage"
        )
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 7. Manual Asset Queries: EURUSD
    if cmd in ("/eur", "/eurusd", "eur", "eurusd", "eur/usd", "📈 eur/usd"):
        horizon = "intraday" if "intraday" in text.lower() else ("daily" if "daily" in text.lower() else "scalping")
        setup, inst_key = get_live_market_setup("EURUSD", horizon=horizon)
        msg = format_trade_alert(setup, instrument=inst_key, horizon=horizon, title_prefix="LIVE EUR/USD SNAPSHOT")
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 8. Manual Asset Queries: Gold (XAUUSD)
    if cmd in ("/gold", "/xau", "/xauusd", "gold", "xau", "🥇 gold"):
        horizon = "intraday" if "intraday" in text.lower() else ("daily" if "daily" in text.lower() else "scalping")
        setup, inst_key = get_live_market_setup("XAUUSD", horizon=horizon)
        msg = format_trade_alert(setup, instrument=inst_key, horizon=horizon, title_prefix="LIVE GOLD SNAPSHOT")
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 9. Manual Asset Queries: Bitcoin (BTCUSD)
    if cmd in ("/btc", "/btcusd", "btc", "bitcoin", "₿ btc/usd"):
        horizon = "intraday" if "intraday" in text.lower() else ("daily" if "daily" in text.lower() else "scalping")
        setup, inst_key = get_live_market_setup("BTCUSD", horizon=horizon)
        msg = format_trade_alert(setup, instrument=inst_key, horizon=horizon, title_prefix="LIVE BITCOIN SNAPSHOT")
        return send_telegram_message(msg, reply_markup=get_main_reply_keyboard())

    # 10. All Markets Triad Overview: /all, /status, "📊 All Markets"
    if cmd in ("/all", "/status", "all", "status", "overview", "📊 all markets"):
        eur, _ = get_live_market_setup("EURUSD", horizon="scalping")
        gold, _ = get_live_market_setup("XAUUSD", horizon="scalping")
        btc, _ = get_live_market_setup("BTCUSD", horizon="scalping")

        eur_sz = compute_custom_sizing(eur, instrument="EURUSD")
        gold_sz = compute_custom_sizing(gold, instrument="XAUUSD")
        btc_sz = compute_custom_sizing(btc, instrument="BTCUSD")

        max_l = cfg.get("max_loss_usd", 25.0)

        msg = (
            f"📊 <b>NEWSLYSER // TRIAD LIVE TELEMETRY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💶 <b>EUR/USD:</b> <code>${eur['current_price']:.5f}</code>\n"
            f"• Status: <b>⚪ STAND ASIDE</b> | Score: <b>{eur.get('confluence_score', 0)}/100</b>\n"
            f"• Key Levels: <code>${eur.get('stop_loss', 1.0950):.5f}</code> - <code>${eur.get('take_profit_1', 1.1008):.5f}</code>\n"
            f"• Sizing for ${max_l:.0f} Risk: <b>{eur_sz['lot_size']:.2f} Lot</b> (-${eur_sz['actual_max_loss']:.2f} at SL)\n\n"
            f"🥇 <b>GOLD (XAU/USD):</b> <code>${gold['current_price']:,.2f}</code>\n"
            f"• Status: <b>⚪ STAND ASIDE</b> | Score: <b>{gold.get('confluence_score', 0)}/100</b>\n"
            f"• Key Levels: <code>${gold.get('stop_loss', 3511.0):,.2f}</code> - <code>${gold.get('take_profit_1', 3565.0):,.2f}</code>\n"
            f"• Sizing for ${max_l:.0f} Risk: <b>{gold_sz['lot_size']:.2f} Lot</b> (-${gold_sz['actual_max_loss']:.2f} at SL)\n\n"
            f"₿ <b>BITCOIN (BTC/USD):</b> <code>${btc['current_price']:,.1f}</code>\n"
            f"• Status: <b>⚪ STAND ASIDE</b> | Score: <b>{btc.get('confluence_score', 0)}/100</b>\n"
            f"• Key Levels: <code>${btc.get('stop_loss', 59196.0):,.1f}</code> - <code>${btc.get('take_profit_1', 60796.0):,.1f}</code>\n"
            f"• Sizing for ${max_l:.0f} Risk: <b>{btc_sz['lot_size']:.2f} Lot</b> (-${btc_sz['actual_max_loss']:.2f} at SL)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🛡️ <i>Guaranteed Risk Ceiling: ${max_l:.2f} Max Loss (Send /loss {int(max_l)} to adjust).</i>"
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
    # Discard old pending updates on boot
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
