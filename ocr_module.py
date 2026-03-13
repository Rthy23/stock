# ═══════════════════════════════════════════════════════════════════════════════
# ocr_module.py  —  Image pre-processing + Gemini OCR + manual correction UI
# ═══════════════════════════════════════════════════════════════════════════════
import io
import re
import json
import streamlit as st

_MODULE = "ocr_module"


def _err(func: str, e: Exception) -> str:
    return (f"MODULE_ERROR: [{_MODULE}] | FUNCTION: [{func}] "
            f"| ERROR: {type(e).__name__}: {e}")


# ── Image pre-processing ──────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Grayscale + auto-contrast enhancement to improve OCR accuracy.
    Returns processed image bytes (PNG).  Falls back to original on error.
    """
    try:
        from PIL import Image, ImageOps, ImageFilter
        img = Image.open(io.BytesIO(image_bytes)).convert("L")   # grayscale
        img = ImageOps.autocontrast(img, cutoff=2)               # contrast
        img = img.filter(ImageFilter.SHARPEN)                    # sharpen
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(_err("preprocess_image", e))
        return image_bytes


# ── Gemini Vision OCR ─────────────────────────────────────────────────────────
def ocr_with_gemini(image_bytes: bytes, api_key: str) -> tuple[list[dict], str | None]:
    """
    Pre-process image then call Gemini Vision to extract fund allocations.

    Returns
    -------
    (items, error_msg)
      items     : list of {"fund_name": str, "percentage": float}
      error_msg : str if failed, None if successful
    """
    try:
        import google.generativeai as genai
        from PIL import Image

        processed = preprocess_image(image_bytes)
        img = Image.open(io.BytesIO(processed))

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are an expert MPF (Mandatory Provident Fund) statement analyzer. "
            "Extract ALL fund names and their percentage allocations from this eMPF screenshot. "
            "Return ONLY a valid JSON array — no markdown, no explanation, no code block. "
            "Format exactly: [{\"fund_name\": \"...\", \"percentage\": 25.5}, ...]. "
            "Round percentage to 1 decimal. "
            "If a value is missing, omit that entry entirely."
        )
        resp = model.generate_content([prompt, img])
        raw  = resp.text.strip()

        cleaned = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        items   = json.loads(cleaned)

        valid = [
            {"fund_name": str(x.get("fund_name", "")).strip(),
             "percentage": float(x.get("percentage", 0))}
            for x in items
            if str(x.get("fund_name", "")).strip()
        ]
        return valid, None

    except json.JSONDecodeError as e:
        return [], f"OCR 解析 JSON 失敗：{e}。請使用手動修正表單。"
    except Exception as e:
        print(_err("ocr_with_gemini", e))
        return [], str(e)


# ── Manual correction form ────────────────────────────────────────────────────
def render_manual_correction_form(
    ocr_items: list[dict],
    form_key:  str = "ocr_correction",
) -> list[dict] | None:
    """
    Show an editable form for the user to correct / confirm OCR results.
    Returns the corrected list on submit, or None if not yet submitted.
    """
    try:
        st.markdown(
            "<div style='background:#1a1a0f; border-left:4px solid #FFD700; "
            "border-radius:6px; padding:10px 14px; margin-bottom:12px;'>"
            "<span style='font-size:13px; color:#FFD700;'>⚠️ OCR 識別結果需要確認 — "
            "請核查並修正下方數據，然後點擊『確認送出』。</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.form(form_key):
            corrected = []
            if ocr_items:
                for i, item in enumerate(ocr_items):
                    c1, c2, c3 = st.columns([4, 2, 1])
                    with c1:
                        name = st.text_input(
                            f"基金名稱 #{i+1}",
                            value=item.get("fund_name", ""),
                            key=f"{form_key}_name_{i}",
                        )
                    with c2:
                        pct = st.number_input(
                            "佔比 %",
                            min_value=0.0, max_value=100.0,
                            value=float(item.get("percentage", 0)),
                            step=0.5,
                            key=f"{form_key}_pct_{i}",
                        )
                    with c3:
                        keep = st.checkbox("保留", value=True,
                                           key=f"{form_key}_keep_{i}")
                    if keep:
                        corrected.append({"fund_name": name.strip(),
                                          "percentage": pct})
            else:
                st.info("OCR 未識別到任何項目。請手動填寫：")
                for i in range(5):
                    c1, c2 = st.columns([4, 2])
                    with c1:
                        name = st.text_input(f"基金名稱 #{i+1}", key=f"{form_key}_mn_{i}")
                    with c2:
                        pct  = st.number_input("佔比 %", min_value=0.0,
                                               max_value=100.0, value=0.0,
                                               step=0.5, key=f"{form_key}_mp_{i}")
                    if name.strip():
                        corrected.append({"fund_name": name.strip(),
                                          "percentage": pct})

            submitted = st.form_submit_button("✅ 確認送出", type="primary")
            if submitted:
                valid = [x for x in corrected if x["fund_name"]]
                return valid if valid else None
        return None
    except Exception as e:
        st.error(f"表單渲染異常：{e}")
        return None


# ── Gemini text generation (for backtest reports) ────────────────────────────
def generate_quant_report(metrics: dict, api_key: str) -> str | None:
    """
    Call Gemini-1.5-flash to generate a brief quantitative analysis paragraph
    from the provided backtest metrics dict.

    Returns the generated text, or None on failure.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        pf   = metrics.get("portfolio_metrics", {})
        bm   = metrics.get("benchmark_metrics", {})
        alpha= metrics.get("alpha")
        tickers = metrics.get("pf_tickers", [])
        years   = metrics.get("window_years", "N/A")
        strategy= metrics.get("strategy_mode", "買入持有")

        def _p(v, mult=100, decimals=2):
            return f"{v*mult:+.{decimals}f}%" if v is not None else "N/A"

        sharpe_str = f"{pf.get('sharpe'):.2f}" if pf.get("sharpe") is not None else "N/A"
        prompt = (
            f"你是一位量化分析師，請用繁體中文，以200字以內，"
            f"根據以下回測數據撰寫一段簡潔的量化分析報告。"
            f"必須點出：(1)策略績效亮點 (2)主要風險點 (3)是否建議使用此策略。\n\n"
            f"回測標的: {', '.join(tickers)}\n"
            f"策略模式: {strategy}\n"
            f"回測年期: {years} 年\n"
            f"組合 CAGR: {_p(pf.get('cagr'))}\n"
            f"組合 Sharpe: {sharpe_str}\n"
            f"最大回撤: {_p(pf.get('max_dd'))}\n"
            f"年化波動: {_p(pf.get('vol'))}\n"
            f"基準 CAGR ({metrics.get('benchmark_ticker','SPY')}): {_p(bm.get('cagr'))}\n"
            f"超額報酬 Alpha: {_p(alpha)}\n"
            f"請輸出純文字，不要使用 Markdown 格式，不要加標題。"
        )
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(_err("generate_quant_report", e))
        return None
