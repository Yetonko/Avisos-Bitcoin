#!/usr/bin/env python3
"""
Bitcoin Metrics Alert Bot — v2
==============================
Métricas on-chain y de derivados para identificar oportunidades de compra/venta.

Fuentes:
  - Puell Multiple  : CoinMetrics Community API (gratuita, sin clave)
  - Resto            : charts.bgeometrics.com (JSONs públicos, sin clave)

Uso:
  python bitcoin_alerts.py           # ejecutar una vez
  python bitcoin_alerts.py --daemon  # ejecutar diariamente a las 08:00
"""

import os
import sys
import time
import logging
import smtplib
import argparse
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
import numpy as np
import pandas as pd
import schedule
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

COINMETRICS_URL  = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BGEOMETRICS_BASE = "https://charts.bgeometrics.com/files"

# ---------------------------------------------------------------------------
# Zonas de señal
# ---------------------------------------------------------------------------

ZONES = {
    "puell": [
        (0.00, 0.50, "COMPRA FUERTE",   "#1a6e3f", "💚"),
        (0.50, 1.00, "Acumulación",     "#2e7d32", "🟢"),
        (1.00, 2.00, "Neutral",         "#616161", "⚪"),
        (2.00, 4.00, "Precaución",      "#e65100", "🟠"),
        (4.00, 999,  "VENTA FUERTE",    "#b71c1c", "🔴"),
    ],
    "mvrv": [
        (-999,  0.0, "COMPRA FUERTE",   "#1a6e3f", "💚"),
        (0.0,   2.0, "Acumulación",     "#2e7d32", "🟢"),
        (2.0,   5.0, "Precaución",      "#e65100", "🟠"),
        (5.0,   7.0, "Zona de Venta",   "#c62828", "🔴"),
        (7.0,   999, "VENTA FUERTE",    "#b71c1c", "🔴"),
    ],
    "nupl": [
        (-999,  0.00, "COMPRA FUERTE",  "#1a6e3f", "💚"),  # Capitulación
        (0.00,  0.25, "Acumulación",    "#2e7d32", "🟢"),  # Esperanza/Miedo
        (0.25,  0.50, "Neutral",        "#616161", "⚪"),  # Optimismo
        (0.50,  0.75, "Precaución",     "#e65100", "🟠"),  # Creencia
        (0.75,  999,  "VENTA FUERTE",   "#b71c1c", "🔴"),  # Euforia
    ],
    "sopr": [
        (-999,  0.970, "COMPRA FUERTE", "#1a6e3f", "💚"),  # Capitulación fuerte
        (0.970, 1.000, "Acumulación",   "#2e7d32", "🟢"),  # Vendiendo a pérdidas
        (1.000, 1.020, "Neutral",       "#616161", "⚪"),
        (1.020, 1.060, "Precaución",    "#e65100", "🟠"),
        (1.060, 999,   "VENTA FUERTE",  "#b71c1c", "🔴"),  # Todo el mundo en ganancias
    ],
    "funding": [
        (-999,    -0.005, "COMPRA FUERTE", "#1a6e3f", "💚"),  # Miedo extremo en derivados
        (-0.005,   0.000, "Acumulación",   "#2e7d32", "🟢"),  # Ligeramente negativo
        (0.000,    0.020, "Neutral",       "#616161", "⚪"),
        (0.020,    0.050, "Precaución",    "#e65100", "🟠"),
        (0.050,    999,   "VENTA FUERTE",  "#b71c1c", "🔴"),  # Codicia extrema
    ],
    "supply": [
        (-999,  50.0, "COMPRA FUERTE",  "#1a6e3f", "💚"),  # Mayoría en pérdidas
        (50.0,  65.0, "Acumulación",    "#2e7d32", "🟢"),
        (65.0,  80.0, "Neutral",        "#616161", "⚪"),
        (80.0,  90.0, "Precaución",     "#e65100", "🟠"),
        (90.0,  999,  "VENTA FUERTE",   "#b71c1c", "🔴"),  # Casi todos en ganancias
    ],
}

# Puntuación por zona para el score compuesto
ZONE_SCORE = {
    "COMPRA FUERTE":   2,
    "Acumulación":     1,
    "Neutral-Alcista": 1,
    "Neutral":         0,
    "Precaución":     -1,
    "Zona de Venta":  -1,
    "VENTA FUERTE":   -2,
}


def _zone_info(value: float, metric: str) -> dict:
    for low, high, label, color, emoji in ZONES[metric]:
        if low <= value < high:
            return {"label": label, "color": color, "emoji": emoji}
    return {"label": "Desconocido", "color": "#9e9e9e", "emoji": "❓"}


# ---------------------------------------------------------------------------
# Obtención de datos — CoinMetrics (Puell Multiple)
# ---------------------------------------------------------------------------

def _fetch_coinmetrics(metrics: list, days: int) -> pd.DataFrame:
    end   = datetime.utcnow()
    start = end - timedelta(days=days)
    params = {
        "assets":     "btc",
        "metrics":    ",".join(metrics),
        "start_time": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end_time":   end.strftime("%Y-%m-%dT23:59:59Z"),
        "frequency":  "1d",
        "page_size":  days + 10,
    }
    log.info("CoinMetrics: descargando %s", metrics)
    resp = requests.get(COINMETRICS_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise ValueError("CoinMetrics no devolvió datos.")
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def get_puell_multiple() -> dict:
    df = _fetch_coinmetrics(["IssTotUSD"], days=400)
    df.dropna(subset=["IssTotUSD"], inplace=True)
    df["MA365"] = df["IssTotUSD"].rolling(365, min_periods=200).mean()
    df.dropna(subset=["MA365"], inplace=True)

    latest   = df.iloc[-1]
    puell    = float(latest["IssTotUSD"] / latest["MA365"])
    issuance = float(latest["IssTotUSD"])
    ma365    = float(latest["MA365"])
    date_str = latest.name.strftime("%d/%m/%Y")

    return {
        "value":    puell,
        "issuance": issuance,
        "ma365":    ma365,
        "date":     date_str,
        "zone":     _zone_info(puell, "puell"),
    }


# ---------------------------------------------------------------------------
# Obtención de datos — BGeometrics (JSONs públicos)
# ---------------------------------------------------------------------------

def _fetch_bg(filename: str) -> list:
    """Descarga un JSON de bgeometrics y devuelve la lista [[ts_ms, value], ...]."""
    url = f"{BGEOMETRICS_BASE}/{filename}"
    log.info("BGeometrics: %s", filename)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _last_value(data: list) -> tuple:
    """Devuelve (timestamp_ms, value) del último registro no nulo."""
    for ts, val in reversed(data):
        if val is not None:
            return ts, float(val)
    raise ValueError("Sin datos válidos")


def get_mvrv_zscore() -> dict:
    data     = _fetch_bg("mvrv_zscore_data.json")
    ts, val  = _last_value(data)
    date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%d/%m/%Y")
    return {
        "value":  val,
        "date":   date_str,
        "zone":   _zone_info(val, "mvrv"),
        "source": "BGeometrics (datos reales)",
    }


def get_nupl() -> dict:
    data     = _fetch_bg("nupl_data.json")
    ts, val  = _last_value(data)
    date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%d/%m/%Y")

    if val < 0:
        cycle = "Capitulación"
    elif val < 0.25:
        cycle = "Esperanza / Miedo"
    elif val < 0.50:
        cycle = "Optimismo"
    elif val < 0.75:
        cycle = "Creencia / Negación"
    else:
        cycle = "Euforia / Codicia"

    return {
        "value": val,
        "cycle": cycle,
        "date":  date_str,
        "zone":  _zone_info(val, "nupl"),
    }


def get_sopr() -> dict:
    data     = _fetch_bg("sopr_7sma.json")
    ts, val  = _last_value(data)
    date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%d/%m/%Y")
    interpretation = (
        "Vendiendo a pérdidas — posible suelo" if val < 1.0
        else "Vendiendo en ganancias — presión vendedora"
    )
    return {
        "value":          val,
        "interpretation": interpretation,
        "date":           date_str,
        "zone":           _zone_info(val, "sopr"),
    }


def get_funding_rate() -> dict:
    data     = _fetch_bg("funding_rate.json")
    ts, val  = _last_value(data)
    date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%d/%m/%Y")
    val_pct  = val * 100  # convertir a %
    return {
        "value": val_pct,
        "date":  date_str,
        "zone":  _zone_info(val_pct, "funding"),
    }


def get_supply_profit() -> dict:
    data     = _fetch_bg("profit_loss.json")
    ts, val  = _last_value(data)
    date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%d/%m/%Y")
    return {
        "value": val,
        "date":  date_str,
        "zone":  _zone_info(val, "supply"),
    }


def get_sth_realized_price() -> dict:
    data     = _fetch_bg("sth_realized_price.json")
    ts, val  = _last_value(data)
    date_str = datetime.utcfromtimestamp(ts / 1000).strftime("%d/%m/%Y")
    return {"value": val, "date": date_str}


def get_btc_price() -> float:
    data = _fetch_bg("mvrv_zscore_btc_price.json")
    _, val = _last_value(data)
    return val


# ---------------------------------------------------------------------------
# Puntuación compuesta (0-10)
# ---------------------------------------------------------------------------

def compute_score(puell: dict, mvrv: dict, nupl: dict,
                  sopr: dict, funding: dict, supply: dict) -> dict:
    """
    Suma puntuaciones por zona de cada métrica.
    Escala: COMPRA FUERTE=+2 … VENTA FUERTE=-2
    Rango total: -12 a +12  →  normalizado a 0-10
    """
    metrics = [
        ("Puell Multiple",   puell["zone"]["label"]),
        ("MVRV Z-Score",     mvrv["zone"]["label"]),
        ("NUPL",             nupl["zone"]["label"]),
        ("SOPR 7SMA",        sopr["zone"]["label"]),
        ("Funding Rate",     funding["zone"]["label"]),
        ("Supply en Profit", supply["zone"]["label"]),
    ]
    raw   = sum(ZONE_SCORE.get(label, 0) for _, label in metrics)
    score = round((raw + 12) / 24 * 10, 1)

    if score >= 8.0:
        level, color, emoji = "OPORTUNIDAD EXCEPCIONAL", "#1a6e3f", "💚"
    elif score >= 6.5:
        level, color, emoji = "ZONA DE COMPRA",          "#2e7d32", "🟢"
    elif score >= 4.5:
        level, color, emoji = "NEUTRAL",                 "#616161", "⚪"
    elif score >= 3.0:
        level, color, emoji = "PRECAUCIÓN",              "#e65100", "🟠"
    else:
        level, color, emoji = "ZONA DE VENTA",           "#b71c1c", "🔴"

    return {
        "score":   score,
        "level":   level,
        "color":   color,
        "emoji":   emoji,
        "details": metrics,
        "raw":     raw,
    }


# ---------------------------------------------------------------------------
# Lógica de envío
# ---------------------------------------------------------------------------

def should_send_email(score_info: dict, puell: dict) -> bool:
    mode = os.getenv("EMAIL_MODE", "SIGNAL").upper()
    if mode == "ALWAYS":
        return True
    return (
        score_info["score"] >= 6.5
        or puell["zone"]["label"] == "VENTA FUERTE"
        or puell["zone"]["label"] == "COMPRA FUERTE"
    )


# ---------------------------------------------------------------------------
# Construcción del email HTML
# ---------------------------------------------------------------------------

def _fmt_usd(value: float) -> str:
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    if value >= 1e6:
        return f"${value/1e6:.2f}M"
    return f"${value:,.0f}"


def _score_bar(score: float, color: str) -> str:
    pct = int(score / 10 * 100)
    return f"""
    <div style="background:#e0e0e0;border-radius:8px;height:18px;width:100%;margin:8px 0 4px;">
      <div style="background:{color};height:18px;border-radius:8px;width:{pct}%;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:#888;">
      <span>0 — Venta</span><span>5 — Neutral</span><span>10 — Compra</span>
    </div>"""


def _puell_alert_banner(puell: dict) -> str:
    """Banner prominente cuando Puell está en zona extrema (compra O venta)."""
    label = puell["zone"]["label"]
    if label == "VENTA FUERTE":
        return f"""
    <div style="background:#b71c1c;border:3px solid #ff1744;border-radius:10px;
                padding:20px 28px;margin-bottom:20px;text-align:center;">
      <div style="font-size:20px;font-weight:900;color:#fff;letter-spacing:1px;">
        ⚠️&nbsp; ALERTA PUELL MULTIPLE — VENTA FUERTE &nbsp;⚠️
      </div>
      <div style="color:#ffcdd2;font-size:14px;margin-top:8px;">
        Puell = <strong style="color:#fff">{puell['value']:.3f}</strong> —
        Zona histórica de techo de ciclo.<br>
        El Puell Multiple en esta zona ha marcado los máximos de todos los ciclos anteriores.
        <strong style="color:#fff">Señal de alta fiabilidad.</strong>
      </div>
    </div>"""
    elif label == "COMPRA FUERTE":
        return f"""
    <div style="background:#1a6e3f;border:3px solid #00e676;border-radius:10px;
                padding:20px 28px;margin-bottom:20px;text-align:center;">
      <div style="font-size:20px;font-weight:900;color:#fff;letter-spacing:1px;">
        💚&nbsp; SEÑAL PUELL MULTIPLE — COMPRA FUERTE &nbsp;💚
      </div>
      <div style="color:#c8e6c9;font-size:14px;margin-top:8px;">
        Puell = <strong style="color:#fff">{puell['value']:.3f}</strong> —
        Zona histórica de suelo de ciclo.<br>
        El Puell Multiple en esta zona ha marcado los mínimos de todos los ciclos anteriores.
        <strong style="color:#fff">Señal de alta fiabilidad.</strong>
      </div>
    </div>"""
    return ""


def _metric_card(title: str, value_str: str, zone: dict,
                 detail_rows: list, note: str = "") -> str:
    rows_html = "".join(
        f"<tr><td style='padding:3px 10px 3px 0;color:#666;font-size:12px'>{k}</td>"
        f"<td style='padding:3px 0;font-size:12px;font-weight:600'>{v}</td></tr>"
        for k, v in detail_rows
    )
    note_html = (
        f"<div style='font-size:11px;color:#888;margin-top:8px;font-style:italic'>{note}</div>"
        if note else ""
    )
    return f"""
    <div style="background:#fff;border-radius:10px;padding:16px 18px;
                box-shadow:0 2px 8px rgba(0,0,0,.06);">
      <div style="font-size:13px;font-weight:700;color:#1a2f5a;margin-bottom:6px">{title}</div>
      <div style="font-size:26px;font-weight:800;color:{zone['color']};
                  margin-bottom:4px;line-height:1.1">{value_str}</div>
      <div style="display:inline-block;background:{zone['color']};color:#fff;
                  border-radius:14px;padding:2px 10px;font-size:11px;
                  font-weight:700;margin-bottom:8px">
        {zone['emoji']} {zone['label']}
      </div>
      <table style="border-collapse:collapse;width:100%">{rows_html}</table>
      {note_html}
    </div>"""


def _zone_guide(metric_key: str) -> str:
    rows = ""
    for low, high, label, color, emoji in ZONES[metric_key]:
        hi_str = "∞"  if high >= 999  else str(high)
        lo_str = "-∞" if low  <= -999 else str(low)
        rows += (
            f"<tr><td style='padding:2px 6px;font-size:10px;color:#666'>{lo_str}→{hi_str}</td>"
            f"<td style='padding:2px 4px'>"
            f"<span style='background:{color};color:#fff;border-radius:6px;"
            f"padding:1px 7px;font-size:10px'>{emoji} {label}</span></td></tr>"
        )
    return f"<table style='border-collapse:collapse'>{rows}</table>"


def build_email(puell: dict, mvrv: dict, nupl: dict, sopr: dict,
                funding: dict, supply: dict, sth: dict,
                btc_price: float, score_info: dict) -> tuple:

    now      = datetime.now().strftime("%d/%m/%Y %H:%M")
    sth_val  = sth["value"]
    sth_diff = (btc_price - sth_val) / sth_val * 100
    if btc_price < sth_val:
        sth_signal = f"BTC BAJO el coste medio reciente ({sth_diff:.1f}%) — zona de acumulación"
        sth_color  = "#00e676"
    else:
        sth_signal = f"BTC sobre el coste medio reciente (+{sth_diff:.1f}%)"
        sth_color  = "#a0b0cc"

    subject = (
        f"[Bitcoin] {score_info['emoji']} {score_info['level']} "
        f"({score_info['score']:.1f}/10) — "
        f"Puell {puell['value']:.2f} | MVRV {mvrv['value']:.2f} | "
        f"NUPL {nupl['value']:.2f} — {now}"
    )

    puell_banner = _puell_alert_banner(puell)

    card_puell = _metric_card(
        "🔶 Puell Multiple",
        f"{puell['value']:.3f}",
        puell["zone"],
        [
            ("Issuance diaria",  _fmt_usd(puell["issuance"])),
            ("Media móvil 365d", _fmt_usd(puell["ma365"])),
            ("Dato de",          puell["date"]),
        ],
        "El más fiable para identificar techos (&gt;4) y suelos (&lt;0.5) de ciclo.",
    )

    card_mvrv = _metric_card(
        "MVRV Z-Score",
        f"{mvrv['value']:.3f}",
        mvrv["zone"],
        [("Dato de", mvrv["date"])],
        "Mide si el precio está muy por encima o debajo de su valor justo histórico.",
    )

    card_nupl = _metric_card(
        "NUPL",
        f"{nupl['value']:.3f}",
        nupl["zone"],
        [("Fase del ciclo", nupl["cycle"]), ("Dato de", nupl["date"])],
        "Ganancia/pérdida latente agregada del mercado. &lt;0 = capitulación.",
    )

    card_sopr = _metric_card(
        "SOPR (7 SMA)",
        f"{sopr['value']:.4f}",
        sopr["zone"],
        [("Lectura", sopr["interpretation"]), ("Dato de", sopr["date"])],
        "&lt;1 = el mercado vende a pérdidas. Señal de capitulación / posible suelo.",
    )

    card_funding = _metric_card(
        "Funding Rate",
        f"{funding['value']:.4f}%",
        funding["zone"],
        [("Dato de", funding["date"])],
        "Negativo = posiciones cortas dominan los futuros. Señal alcista.",
    )

    card_supply = _metric_card(
        "% Supply en Ganancias",
        f"{supply['value']:.1f}%",
        supply["zone"],
        [("Dato de", supply["date"])],
        "&lt;50% = mayoría de BTC en pérdidas. Zona histórica de capitulación.",
    )

    # Detalle del score
    score_rows = "".join(
        f"<tr>"
        f"<td style='padding:3px 10px 3px 0;font-size:12px;color:#555'>{name}</td>"
        f"<td style='padding:3px 8px 3px 0;font-size:12px;font-weight:700;"
        f"color:{'#1a6e3f' if ZONE_SCORE.get(label,0)>0 else '#b71c1c' if ZONE_SCORE.get(label,0)<0 else '#616161'}'>"
        f"{'+' if ZONE_SCORE.get(label,0)>0 else ''}{ZONE_SCORE.get(label,0)}</td>"
        f"<td style='padding:3px 0;font-size:12px;color:#333'>{label}</td>"
        f"</tr>"
        for name, label in score_info["details"]
    )

    guides_html = "".join(
        f"""<div>
          <div style="font-size:11px;font-weight:700;color:#555;margin-bottom:4px">{name}</div>
          {_zone_guide(key)}
        </div>"""
        for name, key in [
            ("Puell Multiple", "puell"),
            ("MVRV Z-Score",   "mvrv"),
            ("NUPL",           "nupl"),
            ("SOPR 7SMA",      "sopr"),
            ("Funding Rate",   "funding"),
            ("Supply Profit",  "supply"),
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:Arial,sans-serif;">
<div style="max-width:640px;margin:30px auto;padding:0 12px;">

  <!-- CABECERA -->
  <div style="background:linear-gradient(135deg,#1a2f5a,#243d73);
              border-radius:12px 12px 0 0;padding:24px 28px;">
    <div style="display:flex;justify-content:space-between;align-items:center;
                flex-wrap:wrap;gap:8px;">
      <div style="font-size:24px;font-weight:800;color:#c9a84c;letter-spacing:1px;">
        ₿ Bitcoin Metrics
      </div>
      <div style="text-align:right">
        <div style="color:#fff;font-size:24px;font-weight:700">{_fmt_usd(btc_price)}</div>
        <div style="color:#a0b0cc;font-size:11px">{now}</div>
      </div>
    </div>
    <div style="margin-top:12px;padding:10px 14px;background:rgba(255,255,255,.09);
                border-radius:8px;font-size:12px;color:{sth_color};">
      <strong>STH Realized Price:</strong> {_fmt_usd(sth_val)}
      &nbsp;·&nbsp; {sth_signal}
    </div>
  </div>

  <div style="background:#f8f7f4;padding:20px 16px 24px;">

    <!-- BANNER PUELL (solo zona extrema) -->
    {puell_banner}

    <!-- PUNTUACIÓN COMPUESTA -->
    <div style="background:#fff;border-radius:10px;padding:20px 24px;
                margin-bottom:18px;box-shadow:0 2px 8px rgba(0,0,0,.06);">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  flex-wrap:wrap;gap:8px;margin-bottom:4px;">
        <div>
          <div style="font-size:12px;color:#888;margin-bottom:2px">
            Puntuación de Oportunidad
          </div>
          <div style="font-size:34px;font-weight:900;color:{score_info['color']};line-height:1">
            {score_info['score']:.1f}
            <span style="font-size:16px;color:#bbb;font-weight:400">/ 10</span>
          </div>
        </div>
        <div style="background:{score_info['color']};color:#fff;border-radius:24px;
                    padding:10px 20px;font-size:15px;font-weight:800;text-align:center;">
          {score_info['emoji']}&nbsp; {score_info['level']}
        </div>
      </div>
      {_score_bar(score_info['score'], score_info['color'])}
      <table style="border-collapse:collapse;width:100%;margin-top:12px;">
        <tr>
          <th style="text-align:left;font-size:11px;color:#aaa;padding-bottom:4px">Métrica</th>
          <th style="text-align:left;font-size:11px;color:#aaa;padding-bottom:4px">Pts</th>
          <th style="text-align:left;font-size:11px;color:#aaa;padding-bottom:4px">Zona</th>
        </tr>
        {score_rows}
      </table>
    </div>

    <!-- GRID DE MÉTRICAS (2 columnas) -->
    <table style="width:100%;border-collapse:separate;border-spacing:10px;">
      <tr>
        <td style="width:50%;vertical-align:top;padding:0">{card_puell}</td>
        <td style="width:50%;vertical-align:top;padding:0">{card_mvrv}</td>
      </tr>
      <tr>
        <td style="vertical-align:top;padding:0">{card_nupl}</td>
        <td style="vertical-align:top;padding:0">{card_sopr}</td>
      </tr>
      <tr>
        <td style="vertical-align:top;padding:0">{card_funding}</td>
        <td style="vertical-align:top;padding:0">{card_supply}</td>
      </tr>
    </table>

    <!-- GUÍA DE ZONAS -->
    <div style="background:#fff;border-radius:10px;padding:18px 20px;
                box-shadow:0 2px 8px rgba(0,0,0,.06);margin-top:10px;">
      <div style="font-size:12px;font-weight:700;color:#1a2f5a;margin-bottom:12px;">
        Guía de interpretación
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
        {guides_html}
      </div>
    </div>

  </div>

  <!-- PIE -->
  <div style="background:#1a2f5a;border-radius:0 0 12px 12px;
              padding:14px 20px;text-align:center;">
    <div style="font-size:11px;color:#7a8faa;line-height:1.8">
      Puell: <a href="https://community-api.coinmetrics.io" style="color:#c9a84c">CoinMetrics</a>
      &nbsp;·&nbsp;
      Resto: <a href="https://charts.bgeometrics.com" style="color:#c9a84c">BGeometrics</a>
      &nbsp;·&nbsp;
      Ref: <a href="https://alfabitcoin.io/indicadores-premium" style="color:#c9a84c">AlfaBitcoin</a><br>
      Análisis meramente informativo. No constituye asesoramiento financiero.
    </div>
  </div>

</div>
</body></html>"""

    return subject, html


# ---------------------------------------------------------------------------
# Envío de email
# ---------------------------------------------------------------------------

def send_email(subject: str, html_body: str) -> bool:
    smtp_host  = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port  = int(os.getenv("SMTP_PORT", "587"))
    smtp_user  = os.getenv("SMTP_USER", "")
    smtp_pass  = os.getenv("SMTP_PASSWORD", "")
    dest_email = os.getenv("ALERT_EMAIL", "")

    if not all([smtp_user, smtp_pass, dest_email]):
        log.error("Faltan credenciales. Revisa SMTP_USER, SMTP_PASSWORD y ALERT_EMAIL en .env")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Bitcoin Alerts <{smtp_user}>"
    msg["To"]      = dest_email
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [dest_email], msg.as_string())
        log.info("Email enviado a %s", dest_email)
        return True
    except smtplib.SMTPAuthenticationError:
        log.error(
            "Error de autenticación SMTP. "
            "Para Gmail usa una Contraseña de Aplicación: "
            "https://myaccount.google.com/apppasswords"
        )
    except Exception as exc:
        log.error("Error al enviar email: %s", exc)
    return False


# ---------------------------------------------------------------------------
# Tarea principal
# ---------------------------------------------------------------------------

def run_check():
    log.info("=== Iniciando revisión de métricas Bitcoin ===")
    try:
        puell   = get_puell_multiple()
        mvrv    = get_mvrv_zscore()
        nupl    = get_nupl()
        sopr    = get_sopr()
        funding = get_funding_rate()
        supply  = get_supply_profit()
        sth     = get_sth_realized_price()
        price   = get_btc_price()

        score_info = compute_score(puell, mvrv, nupl, sopr, funding, supply)

        log.info("BTC Price     : %s",      _fmt_usd(price))
        log.info("STH Realizado : %s",      _fmt_usd(sth["value"]))
        log.info("Puell Multiple: %.3f  [%s]",  puell["value"],    puell["zone"]["label"])
        log.info("MVRV Z-Score  : %.3f  [%s]",  mvrv["value"],     mvrv["zone"]["label"])
        log.info("NUPL          : %.3f  [%s]",  nupl["value"],     nupl["zone"]["label"])
        log.info("SOPR 7SMA     : %.4f  [%s]",  sopr["value"],     sopr["zone"]["label"])
        log.info("Funding Rate  : %.4f%% [%s]", funding["value"],  funding["zone"]["label"])
        log.info("Supply Profit : %.1f%%  [%s]", supply["value"],  supply["zone"]["label"])
        log.info("Score         : %.1f/10  → %s", score_info["score"], score_info["level"])

        if should_send_email(score_info, puell):
            subject, html = build_email(
                puell, mvrv, nupl, sopr, funding, supply, sth, price, score_info
            )
            send_email(subject, html)
        else:
            log.info(
                "Score %.1f (%s) — sin señal activa. "
                "Usa EMAIL_MODE=ALWAYS para recibir siempre el informe.",
                score_info["score"], score_info["level"],
            )

    except requests.exceptions.RequestException as exc:
        log.error("Error de red: %s", exc)
    except Exception as exc:
        log.exception("Error inesperado: %s", exc)
    log.info("=== Revisión completada ===\n")


def main():
    parser = argparse.ArgumentParser(description="Bitcoin Metrics Alert Bot v2")
    parser.add_argument("--daemon", action="store_true",
                        help="Ejecutar en modo demonio (comprueba diariamente a la hora indicada)")
    parser.add_argument("--time", type=str, default="07:30",
                        help="Hora de ejecución diaria en modo demonio, formato HH:MM (por defecto: 07:30)")
    args = parser.parse_args()

    if args.daemon:
        time_str = args.time
        log.info("Modo demonio activo. Revisión diaria a las %s (hora local).", time_str)
        run_check()
        schedule.every().day.at(time_str).do(run_check)
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_check()


if __name__ == "__main__":
    main()
