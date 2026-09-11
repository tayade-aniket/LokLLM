from __future__ import annotations
import time
from typing import Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)

_DEMO_RESPONSES = {
    "default": "[DEMO] This is a simulated response. No LLM is loaded on this device. In a full deployment, the model would generate a personalized response based on your local training data.",
    "healthcare": "[DEMO] Based on your health profile, I recommend consulting a qualified healthcare professional. This response is personalized to your local healthcare context.",
    "education": "[DEMO] Here is a learning tip tailored to your educational context. Practice consistently and seek help when needed.",
    "financial": "[DEMO] This financial guidance is personalized to your local context. Always consult a certified financial advisor for major decisions.",
}


def _detect_domain(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ["health", "doctor", "medicine", "fever", "ill", "hospital", "बुखार", "दवा"]):
        return "healthcare"
    if any(w in prompt_lower for w in ["study", "learn", "school", "exam", "education", "शिक्षा", "शिकण"]):
        return "education"
    if any(w in prompt_lower for w in ["money", "finance", "invest", "bank", "loan", "சேமிப்பு", "முதலீடு"]):
        return "financial"
    return "default"


def generate_demo_response(prompt: str, domain: Optional[str] = None) -> dict:
    if domain is None:
        domain = _detect_domain(prompt)
    response = _DEMO_RESPONSES.get(domain, _DEMO_RESPONSES["default"])
    return {
        "response": response,
        "mode": "DEMO",
        "domain": domain,
        "latency_ms": 0,
        "model_id": None,
    }


def generate_response(
    prompt: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    domain: Optional[str] = None,
) -> dict:
    if model is None or tokenizer is None:
        return generate_demo_response(prompt, domain)

    start = time.perf_counter()

    try:
        import torch

        formatted_prompt = f"### Input:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=64,
            truncation=True,
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "response": response_text,
            "mode": "LIGHTWEIGHT_LOCAL",
            "domain": domain or _detect_domain(prompt),
            "latency_ms": latency_ms,
            "model_id": None,
        }

    except Exception as exc:
        logger.warning(f"Inference failed, switching to DEMO mode: {exc}")
        return generate_demo_response(prompt, domain)


def generate_base_and_personalized(
    prompt: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    personalization_context: Optional[str] = None,
    domain: Optional[str] = None,
) -> dict:
    base = generate_response(prompt, model, tokenizer, domain=domain)

    if personalization_context:
        personalized_prompt = f"Context: {personalization_context}\n\n{prompt}"
    else:
        personalized_prompt = prompt

    personalized = generate_response(personalized_prompt, model, tokenizer, domain=domain)

    if base["mode"] == "DEMO":
        personalized["response"] = (
            "[DEMO] Personalized response: After local fine-tuning on your private data, "
            "the model adapts its language and context to your domain. "
            "Raw training data never leaves your device."
        )
        personalized["mode"] = "DEMO"

    return {
        "prompt": prompt,
        "base_response": base,
        "personalized_response": personalized,
        "personalization_applied": personalization_context is not None or base["mode"] == "DEMO",
    }
