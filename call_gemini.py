"""
call_gemini.py (v1.2)
---------------------
Integração híbrida:
- Usa Google Gemini 1.5 Flash se GEMINI_API_KEY estiver definida.
- Caso contrário, retorna uma resposta simulada (stub) para desenvolvimento local.
"""

import os
import textwrap

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"[Stub LLM] Resposta simulada (modo offline) para o prompt: {prompt[:300]}..."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return textwrap.shorten(response.text.strip(), width=1200, placeholder="...")
    except Exception as e:
        return f"[Stub Fallback] Erro ao chamar LLM real: {str(e)[:150]}"
