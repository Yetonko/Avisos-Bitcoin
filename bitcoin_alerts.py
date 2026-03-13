#!/usr/bin/env python3
"""
Bitcoin Metrics Alert Bot
=========================
Obtiene métricas on-chain de Bitcoin (Puell Multiple y MVRV Z-Score)
usando la API pública de CoinMetrics y envía alertas por email.

Fuentes de referencia:
  - Puell Multiple : https://www.bitcoinmagazinepro.com/es/charts/puell-multiple/
  - MVRV Z-Score   : https://www.bitcoinmagazinepro.com/es/charts/mvrv-zscore/

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

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# CoinMetrics Community API (gratuita, sin clave)
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

# Zonas de señal con descripción y color HTML
ZONES = {
    "puell": [
        (0.00, 0.50, "COMPRA FUERTE",  "#1a6e3f", "💚"),
        (0.50, 1.00, "Acumulación",    "#2e7d32", "🟢"),
        (1.00, 2.00, "Neutral",        "#616161", "⚪"),
        (2.00, 4.00, "Precaución",     "#e65100", "🟠"),
        (4.00, 999,  "VENTA FUERTE",   "#b71c1c", "🔴"),
    ],
    "mvrv": [
        (-999,  0.0, "COMPRA FUERTE",  "#1a6e3f", "💚"),
        (0.0,   2.0, "Neutral-Alcista","#2e7d32", "🟢"),
        (2.0,   5.0, "Precaución",     "#e65100", "🟠"),
        (5.0,   7.0, "Zona de Venta",  "#c62828", "🔴"),
        (7.0,   999, "VENTA FUERTE",   "#b71c1c", "🔴"),
    ],
}


def _zone_info(value: float, metric: str) -> dict:
    """Devuelve descripción, color y emoji para el valor según el indicador."""
    for low, high, label, color, emoji in ZONES[metric]:
        if low <= value < high:
            return {"label": label, "color": color, "emoji": emoji}
    return {"label": "Desconocido", "color": "#9e9e9e", "emoji": "❓"}


# ---------------------------------------------------------------------------
# Obtención de datos — CoinMetrics Community API
# ---------------------------------------------------------------------------

def _fetch_metrics(metrics: list[str], days: int) -> pd.DataFrame:
    """Descarga métricas de CoinMetrics para BTC."""
    end   = datetime.utcnow()
    start = end - timedelta(days=days)
    params = {
        "assets":      "btc",
        "metrics":     ",".join(metrics),
        "start_time":  start.strftime("%Y-%m-%dT00:00:00Z"),
        "end_time":    end.strftime("%Y-%m-%dT23:59:59Z"),
        "frequency":   "1d",
        "page_size":   days + 10,
    }
    log.info("Descargando métricas: %s", metrics)
    resp = requests.get(COINMETRICS_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise ValueError("La API no devolvió datos.")
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def get_puell_multiple() -> dict:
    """
    Puell Multiple = Issuance diaria USD / Media móvil 365d de Issuance USD

    IssTotUSD: valor total en USD de las monedas emitidas ese día.
    Se necesitan ~400 días para calcular la MA-365 con margen.
    """
    df = _fetch_metrics(["IssTotUSD"], days=400)
    df.dropna(subset=["IssTotUSD"], inplace=True)

    df["MA365"] = df["IssTotUSD"].rolling(365, min_periods=200).mean()
    df.dropna(subset=["MA365"], inplace=True)

    latest     = df.iloc[-1]
    puell      = float(latest["IssTotUSD"] / latest["MA365"])
    issuance   = float(latest["IssTotUSD"])
    ma365      = float(latest["MA365"])
    date_str   = latest.name.strftime("%d/%m/%Y")

    return {
        "value":    puell,
        "issuance": issuance,
        "ma365":    ma365,
        "date":     date_str,
        "zone":     _zone_info(puell, "puell"),
    }


def get_mvrv_zscore() -> dict:
    """
    MVRV Z-Score = (Market Cap − Realized Cap) / σ(Market Cap − Realized Cap)

    Se utilizan todos los datos históricos disponibles para σ (~10 años).
    CapMrktCurUSD : Market Cap actual
    CapRealUSD    : Realized Cap (precio al que cada moneda fue movida por última vez)
    """
    df = _fetch_metrics(["CapMrktCurUSD", "CapRealUSD"], days=4000)
    df.dropna(subset=["CapMrktCurUSD", "CapRealUSD"], inplace=True)

    df["diff"]   = df["CapMrktCurUSD"] - df["CapRealUSD"]
    std_dev      = float(df["diff"].std())
    latest       = df.iloc[-1]
    z_score      = float(latest["diff"] / std_dev) if std_dev else 0.0
    mvrv_ratio   = float(latest["CapMrktCurUSD"] / latest["CapRealUSD"])
    date_str     = latest.name.strftime("%d/%m/%Y")

    return {
        "value":       z_score,
        "mvrv_ratio":  mvrv_ratio,
        "market_cap":  float(latest["CapMrktCurUSD"]),
        "realized_cap":float(latest["CapRealUSD"]),
        "date":        date_str,
        "zone":        _zone_info(z_score, "mvrv"),
    }


# ---------------------------------------------------------------------------
# Lógica de señales
# ---------------------------------------------------------------------------

def evaluate_signals(puell: dict, mvrv: dict) -> dict:
    """Determina la señal global y el resumen del análisis."""
    buy_signals  = 0
    sell_signals = 0

    for zone_label in [puell["zone"]["label"], mvrv["zone"]["label"]]:
        if "COMPRA" in zone_label or "Acumulación" in zone_label or "Alcista" in zone_label:
            buy_signals += 1
        elif "VENTA" in zone_label or "Precaución" in zone_label:
            sell_signals += 1

    if buy_signals == 2:
        signal = "COMPRA"
        signal_emoji = "💚"
        signal_color = "#1a6e3f"
    elif sell_signals == 2:
        signal = "VENTA"
        signal_emoji = "🔴"
        signal_color = "#b71c1c"
    elif buy_signals == 1 and sell_signals == 0:
        signal = "TENDENCIA ALCISTA"
        signal_emoji = "🟢"
        signal_color = "#2e7d32"
    elif sell_signals == 1 and buy_signals == 0:
        signal = "PRECAUCIÓN"
        signal_emoji = "🟠"
        signal_color = "#e65100"
    else:
        signal = "NEUTRAL"
        signal_emoji = "⚪"
        signal_color = "#616161"

    return {"signal": signal, "emoji": signal_emoji, "color": signal_color}


def should_send_email(signal_info: dict) -> bool:
    """Decide si se envía el email según EMAIL_MODE."""
    mode = os.getenv("EMAIL_MODE", "SIGNAL").upper()
    if mode == "ALWAYS":
        return True
    return signal_info["signal"] in ("COMPRA", "VENTA")


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


def build_email(puell: dict, mvrv: dict, signal_info: dict) -> tuple[str, str]:
    """Devuelve (asunto, cuerpo HTML) del email."""
    now      = datetime.now().strftime("%d/%m/%Y %H:%M")
    subject  = (
        f"[Bitcoin] {signal_info['emoji']} {signal_info['signal']} — "
        f"Puell {puell['value']:.2f} | MVRV Z {mvrv['value']:.2f} — {now}"
    )

    def metric_card(title, value_str, zone, detail_rows, ref_url):
        rows_html = "".join(
            f"<tr><td style='padding:4px 12px 4px 0;color:#666;font-size:13px'>{k}</td>"
            f"<td style='padding:4px 0;font-size:13px;font-weight:600'>{v}</td></tr>"
            for k, v in detail_rows
        )
        return f"""
        <div style="background:#fff;border-radius:10px;padding:20px 24px;
                    margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06);">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      flex-wrap:wrap;gap:8px;margin-bottom:14px;">
            <span style="font-size:16px;font-weight:700;color:#1a2f5a">{title}</span>
            <a href="{ref_url}" style="font-size:12px;color:#1a2f5a;text-decoration:none;">
              Ver gráfico ↗
            </a>
          </div>
          <div style="font-size:36px;font-weight:800;color:{zone['color']};
                      margin-bottom:6px;">{value_str}</div>
          <div style="display:inline-block;background:{zone['color']};color:#fff;
                      border-radius:20px;padding:4px 14px;font-size:13px;
                      font-weight:700;margin-bottom:14px;">
            {zone['emoji']} {zone['label']}
          </div>
          <table style="border-collapse:collapse;width:100%">{rows_html}</table>
        </div>"""

    puell_card = metric_card(
        "Puell Multiple",
        f"{puell['value']:.3f}",
        puell["zone"],
        [
            ("Issuance diaria (USD)", _fmt_usd(puell["issuance"])),
            ("Media móvil 365d (USD)", _fmt_usd(puell["ma365"])),
            ("Dato de", puell["date"]),
        ],
        "https://www.bitcoinmagazinepro.com/es/charts/puell-multiple/",
    )

    mvrv_card = metric_card(
        "MVRV Z-Score",
        f"{mvrv['value']:.3f}",
        mvrv["zone"],
        [
            ("Market Cap", _fmt_usd(mvrv["market_cap"])),
            ("Realized Cap", _fmt_usd(mvrv["realized_cap"])),
            ("Ratio MVRV", f"{mvrv['mvrv_ratio']:.2f}x"),
            ("Dato de", mvrv["date"]),
        ],
        "https://www.bitcoinmagazinepro.com/es/charts/mvrv-zscore/",
    )

    def zone_table(metric_key):
        rows = ""
        for low, high, label, color, emoji in ZONES[metric_key]:
            hi_str = "∞" if high >= 999 else str(high)
            lo_str = "-∞" if low <= -999 else str(low)
            rows += (
                f"<tr><td style='padding:4px 8px;font-size:12px;color:#555'>"
                f"{lo_str} → {hi_str}</td>"
                f"<td style='padding:4px 8px'>"
                f"<span style='background:{color};color:#fff;border-radius:10px;"
                f"padding:2px 10px;font-size:12px'>{emoji} {label}</span></td></tr>"
            )
        return f"<table style='border-collapse:collapse'>{rows}</table>"

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:Arial,sans-serif;">
<div style="max-width:600px;margin:30px auto;padding:0 16px;">

  <div style="background:linear-gradient(135deg,#1a2f5a,#243d73);
              border-radius:12px 12px 0 0;padding:28px 32px;text-align:center;">
    <div style="font-size:28px;font-weight:800;color:#c9a84c;
                letter-spacing:1px;">₿ Bitcoin Metrics</div>
    <div style="color:#a0b0cc;font-size:13px;margin-top:6px;">{now}</div>
  </div>

  <div style="background:{signal_info['color']};padding:18px 32px;text-align:center;">
    <div style="font-size:22px;font-weight:800;color:#fff;letter-spacing:1px;">
      {signal_info['emoji']}  SEÑAL GLOBAL: {signal_info['signal']}
    </div>
  </div>

  <div style="background:#f8f7f4;padding:24px 24px 16px;">
    {puell_card}
    {mvrv_card}

    <div style="background:#fff;border-radius:10px;padding:20px 24px;
                box-shadow:0 2px 8px rgba(0,0,0,.06);">
      <div style="font-size:14px;font-weight:700;color:#1a2f5a;margin-bottom:12px;">
        Guía de interpretación
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div>
          <div style="font-size:12px;font-weight:700;color:#555;margin-bottom:6px;">Puell Multiple</div>
          {zone_table("puell")}
        </div>
        <div>
          <div style="font-size:12px;font-weight:700;color:#555;margin-bottom:6px;">MVRV Z-Score</div>
          {zone_table("mvrv")}
        </div>
      </div>
    </div>
  </div>

  <div style="background:#1a2f5a;border-radius:0 0 12px 12px;
              padding:16px 24px;text-align:center;">
    <div style="font-size:11px;color:#7a8faa;line-height:1.7">
      Datos: <a href="https://coinmetrics.io" style="color:#c9a84c;">CoinMetrics</a> ·
      Fuente de referencia:
      <a href="https://www.bitcoinmagazinepro.com/es/charts/puell-multiple/"
         style="color:#c9a84c;">Bitcoin Magazine Pro</a><br>
      Este análisis es meramente informativo. No constituye asesoramiento financiero.
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
        puell = get_puell_multiple()
        log.info("Puell Multiple: %.3f  [%s]", puell["value"], puell["zone"]["label"])

        mvrv = get_mvrv_zscore()
        log.info("MVRV Z-Score : %.3f  [%s]", mvrv["value"], mvrv["zone"]["label"])

        signal_info = evaluate_signals(puell, mvrv)
        log.info("Señal global  : %s %s", signal_info["emoji"], signal_info["signal"])

        if should_send_email(signal_info):
            subject, html = build_email(puell, mvrv, signal_info)
            send_email(subject, html)
        else:
            log.info(
                "Sin señal activa (%s). Email no enviado. "
                "Cambia EMAIL_MODE=ALWAYS para recibir siempre el informe.",
                signal_info["signal"],
            )
    except requests.exceptions.RequestException as exc:
        log.error("Error de red al consultar la API: %s", exc)
    except Exception as exc:
        log.exception("Error inesperado: %s", exc)
    log.info("=== Revisión completada ===\n")


def main():
    parser = argparse.ArgumentParser(description="Bitcoin Metrics Alert Bot")
    parser.add_argument("--daemon", action="store_true",
                        help="Ejecutar en modo demonio (comprueba diariamente a las 08:00)")
    parser.add_argument("--hour", type=int, default=8,
                        help="Hora de ejecución diaria en modo demonio (por defecto: 8)")
    args = parser.parse_args()

    if args.daemon:
        time_str = f"{args.hour:02d}:00"
        log.info("Modo demonio activo. Revisión diaria a las %s.", time_str)
        run_check()
        schedule.every().day.at(time_str).do(run_check)
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_check()


if __name__ == "__main__":
    main()
