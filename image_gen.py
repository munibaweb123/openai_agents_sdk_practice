# save_image_from_responses.py
import os
import re
import base64
import sys

try:
    import requests
except Exception:
    print("Please install requests: pip install requests")
    raise

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()   # loads .env from current working directory
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def write_base64_to_file(b64: str, filename: str):
    data = base64.b64decode(b64)
    with open(filename, "wb") as f:
        f.write(data)

def download_url_to_file(url: str, filename: str):
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in resp.iter_content(1024):
            f.write(chunk)

def extract_image_payload(item):
    """
    Try common shapes where an image payload can appear in a responses output item.
    Returns tuple (kind, value) where kind is 'url'|'base64' or (None, None).
    """
    if item is None:
        return (None, None)

    # item might be a plain string
    if isinstance(item, str):
        s = item.strip()
        # data URI
        m = re.match(r"data:(image/\w+);base64,(.*)", s, re.DOTALL)
        if m:
            return ("base64", m.group(2))
        if s.startswith("http"):
            return ("url", s)
        # maybe plain base64
        if re.fullmatch(r"[A-Za-z0-9+/=\s]+", s) and len(s) % 4 == 0:
            return ("base64", s)
        return (None, None)

    # item might be a dict/obj-like
    if isinstance(item, dict):
        # common keys to check
        for key in ("url", "image_url", "image", "data", "result", "content", "b64", "base64"):
            v = item.get(key)
            if not v:
                continue
            if isinstance(v, str):
                # same checks as above
                s = v.strip()
                m = re.match(r"data:(image/\w+);base64,(.*)", s, re.DOTALL)
                if m:
                    return ("base64", m.group(2))
                if s.startswith("http"):
                    return ("url", s)
                if re.fullmatch(r"[A-Za-z0-9+/=\s]+", s) and len(s) % 4 == 0:
                    return ("base64", s)
            # sometimes nested dict contains url or base64
            if isinstance(v, dict):
                # recursive attempt
                kind, val = extract_image_payload(v)
                if kind:
                    return (kind, val)

    # fallback: try attributes (if response objects behave like objects)
    # many response items are simple dict-like; if not found, return None
    return (None, None)


def save_image_from_response(response, out_filename="generated_image.png"):
    """
    Accepts the response object returned by client.responses.create()
    and tries to find and save an image payload.
    """
    # convert to dict if possible (safe access)
    resp_dict = None
    try:
        resp_dict = response.to_dict()
    except Exception:
        try:
            resp_dict = dict(response)
        except Exception:
            resp_dict = response  # hope it's already dict-like

    outputs = resp_dict.get("output") if isinstance(resp_dict, dict) else None
    if not outputs:
        # sometimes it's under 'choices' or 'output' nested differently
        outputs = resp_dict.get("choices") if isinstance(resp_dict, dict) else None

    # normalize to list
    if isinstance(outputs, dict):
        outputs = [outputs]
    if outputs is None:
        outputs = []

    # scan all items
    for out in outputs:
        # out may itself be a dict with 'type' and 'content'/ 'text'
        # try a few places where the tool-call appears
        candidates = []

        if isinstance(out, dict):
            # some SDKs put top-level fields
            if "type" in out and out["type"] == "image_generation_call":
                candidates.append(out.get("result") or out.get("image") or out.get("content") or out.get("data"))

            # check nested 'content' or 'message' fields
            if "content" in out:
                candidates.append(out["content"])
            if "message" in out:
                candidates.append(out["message"])
            if "result" in out:
                candidates.append(out["result"])
            # tool call structure might be in 'tool_call' or 'tool' keys
            if "tool_call" in out:
                candidates.append(out["tool_call"])
            if "tool" in out:
                candidates.append(out["tool"])
            # sometimes it's a list under 'items' or 'parts'
            if "items" in out:
                candidates.extend(out["items"])
            if "parts" in out:
                candidates.extend(out["parts"])

        # If out isn't dict, try it directly
        candidates.append(out)

        # inspect candidates
        for cand in candidates:
            kind, val = extract_image_payload(cand)
            if kind == "url":
                download_url_to_file(val, out_filename)
                print(f"Saved image from URL -> {out_filename}")
                return True
            if kind == "base64":
                write_base64_to_file(val, out_filename)
                print(f"Saved image from base64 -> {out_filename}")
                return True

    # If we get here, try scanning entire resp_dict for any string that looks like a data URI or URL
    text = str(resp_dict)
    m = re.search(r"(data:image/\w+;base64,[A-Za-z0-9+/=\s]+)", text)
    if m:
        b64_full = m.group(1)
        b64 = re.sub(r"^data:image/\w+;base64,", "", b64_full)
        write_base64_to_file(b64, out_filename)
        print(f"Saved embedded data URI -> {out_filename}")
        return True
    m2 = re.search(r"(https?://[^\s'\"<>]+(?:png|jpg|jpeg|webp))", text)
    if m2:
        download_url_to_file(m2.group(1), out_filename)
        print(f"Saved embedded URL -> {out_filename}")
        return True

    print("No image found in response. Inspect the response object to locate the image payload.")
    return False


if __name__ == "__main__":
    prompt = "Generate an image of gray tabby cat hugging an otter with an orange scarf"

    # Call the Responses API and request the image_generation tool
    response = client.responses.create(
        model="gpt-5",
        input=prompt,
        tools=[{"type": "image_generation"}],
    )

    # try to save image (will print result)
    ok = save_image_from_response(response, out_filename="otter.png")
    if not ok:
        # helpful debug print
        try:
            print("\n--- Response debug (trimmed) ---")
            print(type(response))
            # try printing top-level fields safely
            if hasattr(response, "to_dict"):
                import json
                print(json.dumps(response.to_dict(), indent=2)[:4000])
            else:
                print(str(response)[:4000])
        except Exception as e:
            print("Could not serialize response for debug:", e)
        sys.exit(2)
