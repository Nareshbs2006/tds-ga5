import json, os, urllib.request, urllib.error
from fastapi import FastAPI

app = FastAPI()


@app.get("/debug")
def debug():
    tok = os.environ.get("AIPIPE_TOKEN", "")
    out = {"token_len": len(tok), "token_dots": tok.count(".")}
    try:
        req = urllib.request.Request("https://aipipe.org/openai/v1/embeddings",
                                     data=json.dumps({"model": "text-embedding-3-small", "input": ["hi"]}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok.strip()}", "User-Agent": "Mozilla/5.0 (tds-ga5 server)"})
        with urllib.request.urlopen(req, timeout=10) as r:
            out["status"] = r.status
            out["body"] = r.read().decode()[:120]
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["body"] = e.read().decode()[:300]
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


@app.get("/debug2")
def debug2():
    import index
    out = {}
    try:
        out["emb"] = len(index._embed(["hello"])[0])
    except Exception as e:
        out["emb_err"] = repr(e)[:300]
    try:
        out["llm"] = index._chat("gpt-4.1-mini", "Reply JSON {\"ok\": true}", "hi", timeout=15)
    except urllib.error.HTTPError as e:
        out["llm_err"] = [e.code, e.read().decode()[:300]]
    except Exception as e:
        out["llm_err"] = repr(e)[:300]
    return out
