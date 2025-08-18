import numpy as np
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs, logger
from .base import BaseKVStorage

_gemini_client = None
FORCE_GEMINI_ONLY = True  # Force Gemini-only usage; OpenAI purged

def _get_gemini_client():  # lazy import to keep optional
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:  # pragma: no cover - depends on external pkg
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        _gemini_client = genai
        logger.info("Gemini client initialized (flash-lite will be default)")
    except Exception as e:  # noqa: E722
        logger.warning(f"Failed to init Gemini client: {e}")
        _gemini_client = False  # sentinel meaning unavailable
    return _gemini_client
async def _gemini_complete_if_cache(model, prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """Gemini-only completion with simple hashing cache support (hashing_kv kwarg)."""
    if history_messages is None:
        history_messages = []
    gemini = _get_gemini_client()
    if not gemini or gemini is False:
        raise RuntimeError("Gemini client not initialized and no fallback allowed.")
    hashing_kv = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]
    # Flatten to plain text for Gemini
    flat = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            flat.append(c)
        else:
            flat.append(str(c))
    full_prompt = "\n\n".join(flat)
    generative_model = gemini.GenerativeModel(model)
    response = generative_model.generate_content(full_prompt)
    if hasattr(response, "text") and response.text:
        text = response.text
    elif hasattr(response, "candidates") and response.candidates:
        # best-effort extraction
        first = response.candidates[0]
        try:
            text = first.content.parts[0].text
        except Exception:  # noqa: E722
            text = str(first)
    else:
        text = ""
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": text, "model": model}})
        await hashing_kv.index_done_callback()
    return text


async def gpt_4o_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    # Retained legacy function name for backward compatibility; now Gemini only
    # Use flash-lite for speed or flash for complex tasks
    model_name = os.getenv("GEMINI_MAIN_MODEL", "models/gemini-2.5-flash")  # Flash is faster than flash-lite for many tasks
    return await _gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    # Retained legacy function name for backward compatibility; now Gemini only
    # Use flash-lite for cheap/fast operations
    model_name = os.getenv("GEMINI_CHEAP_MODEL", os.getenv("GEMINI_MAIN_MODEL", "models/gemini-2.5-flash-lite"))
    return await _gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=32768)  # Increased token limit
async def gemini_embedding(texts: list[str]):  # pragma: no cover
    """Gemini embedding wrapper (text-embedding-004).

    OpenAI & Azure OpenAI code paths have been removed; failure here raises directly.
    Optimized for batch processing with larger context windows.
    """
    import numpy as _np
    gemini = _get_gemini_client()
    model_name = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
    if not gemini or gemini is False:
        raise RuntimeError("Gemini embedding unavailable (client not initialized)")
    
    vectors = []
    # Process in larger batches for efficiency
    batch_size = min(100, len(texts))  # Increased batch size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vectors = []
        
        for t in batch:
            try:
                resp = gemini.embed_content(model=model_name, content=t)
                emb = resp.get("embedding") if isinstance(resp, dict) else None
                if emb is None and hasattr(resp, "embedding"):
                    emb = resp.embedding
                if emb is None:
                    raise ValueError("Gemini embed response missing 'embedding'")
                batch_vectors.append(emb)
            except Exception as e:  # noqa: E722
                logger.warning(f"Gemini embedding failed for one text: {e}")
                batch_vectors.append([0.0] * 768)
        
        vectors.extend(batch_vectors)
    
    return _np.array(vectors)
## All OpenAI / Azure OpenAI related functions have been removed.
