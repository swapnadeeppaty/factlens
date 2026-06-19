from dotenv import load_dotenv
load_dotenv()

import os
import requests
import wikipedia
import faiss
import numpy as np
from fastembed import TextEmbedding
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
)
CORS(app)

# ── Gemini setup ───────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    print(f"\n[SYSTEM] API Key loaded: {GEMINI_API_KEY[:8]}...\n")
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("\n[ERROR] GEMINI_API_KEY not found!\n")

# ── Lazy-loaded embedder ───────────────────────────────────────────────────────
embedder = None

def load_models():
    global embedder

    if embedder is None:
        print("[LOAD] Loading FastEmbed model...")
        embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

    print("[LOAD] All models ready.\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_keywords_with_gemini(claim):
    prompt = f"""
Extract the key entities, people, places, and concepts from the following statement to use for a search query.
Return a list of keywords separated by commas. Do not include bullet points or introductory text.
Statement: {claim}
Keywords:"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        keywords = [k.strip() for k in response.text.replace("\n", ",").split(",") if k.strip()]
        print(f"[KEYWORDS] {keywords}")
        return keywords
    except Exception as e:
        print(f"[KEYWORD ERROR] {e}")
        return []


def fetch_wikipedia_summary(query):
    wikipedia.set_user_agent("FactLensApp/1.0 (contact@example.com)")
    try:
        page = wikipedia.page(query, auto_suggest=False)
        text = wikipedia.summary(query, sentences=3, auto_suggest=False)
        print(f"[WIKIPEDIA] OK: {query}")
        return {"text": text, "url": page.url, "title": page.title, "source": "Wikipedia"}
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            choice = e.options[0]
            page = wikipedia.page(choice, auto_suggest=False)
            text = wikipedia.summary(choice, sentences=3, auto_suggest=False)
            print(f"[WIKIPEDIA] Disambiguation -> {choice}")
            return {"text": text, "url": page.url, "title": page.title, "source": "Wikipedia"}
        except:
            return {"text": "", "url": "", "title": query, "source": "Wikipedia"}
    except Exception as e:
        print(f"[WIKIPEDIA] FAIL: {query} -> {e}")
        return {"text": "", "url": "", "title": query, "source": "Wikipedia"}


def fetch_duckduckgo(query):
    """Fetch real-time search result from DuckDuckGo Instant Answer API."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        abstract = data.get("AbstractText", "")
        source_url = data.get("AbstractURL", "")
        if abstract:
            print(f"[DUCKDUCKGO] OK: {query} -> {abstract[:60]}...")
            return {"text": abstract, "url": source_url, "title": "Web Search", "source": "DuckDuckGo"}
        print(f"[DUCKDUCKGO] No result for: {query}")
        return {"text": "", "url": "", "title": "", "source": "DuckDuckGo"}
    except Exception as e:
        print(f"[DUCKDUCKGO ERROR] {e}")
        return {"text": "", "url": "", "title": "", "source": "DuckDuckGo"}


def fetch_wikidata_claim(query):
    """
    Fetch structured data from Wikidata for more up-to-date factual info.
    Good for current officeholders, dates, positions etc.
    """
    try:
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": 1
        }
        r = requests.get(search_url, params=params, timeout=5)
        results = r.json().get("search", [])
        if not results:
            return {"text": "", "url": "", "title": "", "source": "Wikidata"}

        entity_id = results[0]["id"]
        entity_url = f"https://www.wikidata.org/wiki/{entity_id}"
        description = results[0].get("description", "")
        label = results[0].get("label", "")

        if description:
            text = f"{label}: {description}"
            print(f"[WIKIDATA] OK: {query} -> {text}")
            return {"text": text, "url": entity_url, "title": label, "source": "Wikidata"}

        return {"text": "", "url": "", "title": "", "source": "Wikidata"}
    except Exception as e:
        print(f"[WIKIDATA ERROR] {e}")
        return {"text": "", "url": "", "title": "", "source": "Wikidata"}


def build_index(snippets):
    filtered = [s for s in snippets if s["text"] and len(s["text"]) > 20]
    if not filtered:
        raise ValueError("No valid snippets to build index.")

    texts = [s["text"] for s in filtered]
    urls = [s["url"] for s in filtered]
    titles = [s["title"] for s in filtered]
    sources = [s.get("source", "") for s in filtered]

    embs = np.asarray(list(embedder.embed(texts)), dtype="float32")
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)

    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, embs, texts, urls, titles, sources


def semantic_search(query, index, texts, urls, titles, sources, k=5):
    k = min(k, len(texts))
    q_emb = np.asarray(list(embedder.embed([query])), dtype="float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    if np.linalg.norm(q_emb) != 0:
        faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if score > 0:
            results.append({
                "sim": float(score),
                "text": texts[idx],
                "url": urls[idx],
                "title": titles[idx],
                "source": sources[idx],
            })
    return results


def gemini_verdict(claim, evidences):
    evidence_text = "\n".join([
        f"[{e.get('source','?')}] {e['text'][:300]}..." for e in evidences
    ])
    prompt = f"""
Claim: {claim}

Evidence from multiple sources:
{evidence_text}

IMPORTANT INSTRUCTIONS:
- Evidence may include Wikipedia, DuckDuckGo web search, and Wikidata results.
- Political positions, leadership roles, election results, and current officeholders change frequently.
- If the claim is about a current role or recent event and evidence seems outdated or contradictory, classify as UNVERIFIABLE and explain why.
- Prioritize more recent or web-sourced evidence over older encyclopedia entries.
- Be skeptical of claims about "current" status unless evidence clearly confirms it.

Classify the claim strictly using this format (no markdown):
VERDICT: [TRUE, FALSE, or UNVERIFIABLE]
CONFIDENCE: [0-100]
EXPLANATION: [Concise 2-3 sentence explanation]
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        print(f"[VERDICT] {response.text[:100]}...")
        return response.text
    except Exception as e:
        print(f"[VERDICT ERROR] {e}")
        return f"VERDICT: UNVERIFIABLE\nCONFIDENCE: 0\nEXPLANATION: Gemini error — {e}"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/factcheck", methods=["POST"])
def factcheck():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set."}), 500

    data = request.get_json(silent=True) or {}
    claim = (data.get("claim") or "").strip()

    if not claim:
        return jsonify({"error": "No claim provided."}), 400

    try:
        print(f"\n{'='*50}")
        print(f"[CLAIM] {claim}")
        print(f"{'='*50}")

        # Load the lighter model only when the first request arrives
        load_models()

        # Step 1 — extract keywords
        keywords = extract_keywords_with_gemini(claim)
        if not keywords:
            return jsonify({
                "claim": claim,
                "verdict": "VERDICT: UNVERIFIABLE\nCONFIDENCE: 0\nEXPLANATION: Could not extract keywords.",
                "evidence": []
            })

        # Step 2 — fetch from all sources
        snippets = []

        # Wikipedia for each keyword
        for k in keywords:
            snippets.append(fetch_wikipedia_summary(k))

        # DuckDuckGo for the full claim (real-time)
        ddg = fetch_duckduckgo(claim)
        if ddg["text"]:
            snippets.append(ddg)

        # Also DuckDuckGo for the first keyword
        if keywords:
            ddg2 = fetch_duckduckgo(keywords[0])
            if ddg2["text"]:
                snippets.append(ddg2)

        # Wikidata for each keyword (structured current data)
        for k in keywords[:3]:
            wd = fetch_wikidata_claim(k)
            if wd["text"]:
                snippets.append(wd)

        valid_snippets = [s for s in snippets if s["text"] and len(s["text"]) > 20]
        print(f"[SNIPPETS] Total valid: {len(valid_snippets)}")

        if not valid_snippets:
            return jsonify({
                "claim": claim,
                "verdict": "VERDICT: UNVERIFIABLE\nCONFIDENCE: 0\nEXPLANATION: No evidence found.",
                "evidence": []
            })

        # Step 3 — build index and search
        index, embs, texts, urls, titles, sources = build_index(valid_snippets)
        results = semantic_search(claim, index, texts, urls, titles, sources, k=5)
        relevant = [r for r in results if r["sim"] > 0.3]

        # Step 4 — verdict
        if not relevant:
            verdict = "VERDICT: UNVERIFIABLE\nCONFIDENCE: 0\nEXPLANATION: No strongly relevant evidence found."
        else:
            verdict = gemini_verdict(claim, relevant)

        return jsonify({
            "claim": claim,
            "verdict": verdict,
            "evidence": results
        })

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)