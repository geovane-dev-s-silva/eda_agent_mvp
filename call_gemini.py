"""
call_gemini.py
---------------
Módulo híbrido para integração entre o agente EDA e LLM (Gemini / Stub).

- Se a variável de ambiente GEMINI_API_KEY estiver definida, o módulo usa a API real do Google Generative AI.
- Caso contrário, ele retorna uma resposta simulada (modo stub).

Uso:
    from call_gemini import call_gemini
    resposta = call_gemini("Explique a média de vendas por categoria")


"""

import os
import textwrap


def call_gemini(prompt: str) -> str:
    """
    Função híbrida: usa Gemini se chave disponível, senão retorna resposta simulada.
    """
    """
    Função híbrida que usa Gemini se disponível, ou fallback stub.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        # 🔹 Modo desenvolvimento (sem chave)
        return f"[Stub LLM] Resposta simulada para o prompt: {prompt[:300]}..."

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            safety_settings={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            },
        )
        return textwrap.shorten(response.text.strip(), width=1200, placeholder="...")
    except Exception as e:
        # 🔸 Se algo falhar (ex: limite, erro de rede, timeout), caímos pro stub
        return f"[Stub Fallback] Erro ao chamar LLM real: {str(e)[:150]}"
