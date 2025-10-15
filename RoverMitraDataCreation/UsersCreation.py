#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bulk register → login → post preferences for RoverMitra dummy users.

Usage:
  python bulk_register_login_preferences.py \
      --users-json /path/to/users_core.json \
      --base-url https://localhost:7195 \
      --register-endpoint /User/register \
      --login-endpoint /User/login \
      --prefs-endpoint /Preferences \
      --concurrency 4 \
      --verify-ssl false

Notes:
- Expects users_json to be an array of user objects (your 500 dummy users).
- Maps register payload to the fields shown in the Postman collection:
  email, password, confirmPassword, firstName, lastName, middleName,
  phoneNumber, dateOfBirth, address, city, state, postalCode, country. :contentReference[oaicite:1]{index=1}
- Login response parsing is resilient: it tries common keys like "token",
  "access_token", "jwt", etc. Override with --login-token-key if needed.
"""

import os
import json
import time
import argparse
import logging
import random
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry

# ----------------------------- Config & CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bulk register→login→prefs")
    p.add_argument("--users-json", required=True, help="Path to dummy users JSON (array).")
    p.add_argument("--base-url", default=os.getenv("RM_BASE_URL", "https://localhost:7195"),
                   help="API base URL (default: https://localhost:7195)")
    p.add_argument("--register-endpoint", default=os.getenv("RM_REGISTER_EP", "/User/register"),
                   help="Register endpoint path")
    p.add_argument("--login-endpoint", default=os.getenv("RM_LOGIN_EP", "/User/login"),
                   help="Login endpoint path")
    p.add_argument("--prefs-endpoint", default=os.getenv("RM_PREFS_EP", "/Preferences"),
                   help="Preferences endpoint path")
    p.add_argument("--verify-ssl", default=os.getenv("RM_VERIFY_SSL", "false"),
                   choices=["true", "false"], help="Verify TLS certs? (default: false for localhost)")
    p.add_argument("--login-method", default=os.getenv("RM_LOGIN_METHOD", "POST"),
                   choices=["POST", "GET"], help="HTTP method for login (default: POST)")
    p.add_argument("--login-token-key", default=os.getenv("RM_TOKEN_KEY", ""),
                   help="Explicit JSON key for token in login response (optional).")
    p.add_argument("--rate-limit", type=float, default=float(os.getenv("RM_RATE_LIMIT", "0.0")),
                   help="Seconds to sleep between users (default 0.0)")
    p.add_argument("--start", type=int, default=0, help="Start index in users array (default 0)")
    p.add_argument("--limit", type=int, default=0, help="Process at most N users (0 = all)")
    p.add_argument("--timeout", type=float, default=30.0, help="Request timeout seconds (default 30)")
    p.add_argument("--dry-run", action="store_true", help="Don't call APIs; just show what would be sent.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

# ----------------------------- HTTP Session -----------------------------

def make_session(verify_ssl: bool, timeout: float) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "PUT", "PATCH"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.verify = verify_ssl
    s.request = _wrap_request_with_timeout(s.request, timeout)
    if not verify_ssl:
        requests.packages.urllib3.disable_warnings()  # only suppress for local dev
    return s

def _wrap_request_with_timeout(orig_request, timeout):
    def _req(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return orig_request(method, url, **kwargs)
    return _req

# ----------------------------- Helpers -----------------------------

def build_register_payload(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map your rich dummy user object to the register payload as per Postman:
      email, password, confirmPassword, firstName, lastName, middleName,
      phoneNumber, dateOfBirth, address, city, state, postalCode, country. :contentReference[oaicite:2]{index=2}

    If your dummy user object already has these keys, they'll be used directly.
    Otherwise, we fall back to reasonable defaults.
    """
    def pick(*keys, default=""):
        for k in keys:
            v = user.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return default

    email = pick("email", "mail", default=f"user_{random.randint(100000,999999)}@example.com")
    # If your dummy JSON doesn't include plaintext passwords, you can inject one here:
    password = user.get("password") or "Pass@976006"
    first = pick("firstName", "given_name", "first_name", default="Test")
    last  = pick("lastName", "family_name", "last_name", default="User")

    dob = user.get("dateOfBirth", "1999-07-18T00:00:00")
    if isinstance(dob, str) and not dob.endswith("Z"):
        dob = dob + "Z"

    payload = {
        "email": email,
        "password": password,
        "confirmPassword": user.get("confirmPassword", password),
        "firstName": first,
        "lastName": last,
        "middleName": user.get("middleName", ""),
        "phoneNumber": user.get("phoneNumber", "+00-000-000-0000"),
        # Expecting ISO string like Postman example "1999-07-18T00:00:00Z" :contentReference[oaicite:3]{index=3}
        "dateOfBirth": dob,
        "address": user.get("address", "Unknown Street 1"),
        "city": user.get("city", "Unknown City"),
        "state": user.get("state", ""),
        "postalCode": user.get("postalCode", "00000"),
        "country": user.get("country", "Unknown")
    }
    return payload

def extract_token(login_json: Dict[str, Any], explicit_key: Optional[str] = None) -> Optional[str]:
    """Try common token keys, unless an explicit key is supplied."""
    if explicit_key:
        return login_json.get(explicit_key)

    for k in ["token", "access_token", "accessToken", "jwt", "id_token", "bearer", "Bearer"]:
        if k in login_json and isinstance(login_json[k], str) and login_json[k]:
            return login_json[k]

    # Some APIs return { "data": { "token": "..." } }
    data = login_json.get("data")
    if isinstance(data, dict):
        for k in ["token", "access_token", "accessToken", "jwt"]:
            if k in data and isinstance(data[k], str) and data[k]:
                return data[k]
    return None

# ----------------------------- Core flow -----------------------------

def do_register(session: requests.Session, base_url: str, endpoint: str, payload: Dict[str, Any], dry: bool=False) -> requests.Response:
    url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    if dry:
        logging.info(f"[DRY] POST {url} payload={payload}")
        class Dummy: status_code=201; text="dry-run"; 
        return Dummy()  # type: ignore
    resp = session.post(url, json=payload)
    return resp

def do_login(session: requests.Session, base_url: str, endpoint: str,
             email: str, password: str, method: str="POST", dry: bool=False) -> Dict[str, Any]:
    url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    login_payload = {"email": email, "password": password}
    if dry:
        logging.info(f"[DRY] {method} {url} payload={login_payload}")
        return {"token": "dry-token"}
    if method.upper() == "GET":
        r = session.get(url, params=login_payload)
    else:
        r = session.post(url, json=login_payload)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

def post_preferences(session: requests.Session, base_url: str, endpoint: str,
                     token: str, full_user_object: Dict[str, Any], dry: bool=False) -> requests.Response:
    url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    headers = {"Authorization": f"Bearer {token}"}
    if dry:
        logging.info(f"[DRY] POST {url} headers={headers} json=<full_user_object>")
        class Dummy: status_code=200; text="dry-run";
        return Dummy()  # type: ignore
    resp = session.post(url, headers=headers, json=full_user_object)
    return resp

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    verify_ssl = args.verify_ssl.lower() == "true"

    # Load users
    with open(args.users_json, "r", encoding="utf-8") as f:
        users = json.load(f)
    if not isinstance(users, list):
        raise ValueError("users_json must be a JSON array of user objects.")

    end_idx = len(users) if args.limit == 0 else min(len(users), args.start + args.limit)
    users_slice = users[args.start:end_idx]
    logging.info(f"Processing users[{args.start}:{end_idx}] → {len(users_slice)} users")

    session = make_session(verify_ssl=verify_ssl, timeout=args.timeout)

    success = 0
    for i, user in enumerate(users_slice, start=args.start):
        try:
            reg_payload = build_register_payload(user)
            email = reg_payload["email"]
            password = reg_payload["password"]

            # 1) Register
            r = do_register(session, args.base_url, args.register_endpoint, reg_payload, dry=args.dry_run)
            sc = getattr(r, "status_code", 0)
            if sc in (200, 201):
                logging.info(f"[{i}] Register OK for {email} (status {sc})")
            elif sc in (400, 409, 422):
                # Likely 'already exists' or validation. Continue to login anyway.
                logging.warning(f"[{i}] Register non-2xx ({sc}) for {email}: {getattr(r,'text','')[:200]}")
            else:
                logging.warning(f"[{i}] Register unexpected status {sc} for {email}: {getattr(r,'text','')[:200]}")

            # 2) Login
            login_json = do_login(session, args.base_url, args.login_endpoint, email, password,
                                  method=args.login_method, dry=args.dry_run)
            token = extract_token(login_json, explicit_key=args.login_token_key or None)
            if not token:
                logging.error(f"[{i}] Failed to extract token for {email}. Login response: {str(login_json)[:300]}")
                continue

            # 3) Preferences with full user JSON            pr = post_preferences(session, args.base_url, args.prefs_endpoint, token, user, dry=args.dry_run)
            psc = getattr(pr, "status_code", 0)
            if psc in (200, 201):
                logging.info(f"[{i}] Preferences OK for {email} (status {psc})")
                success += 1
            else:
                logging.error(f"[{i}] Preferences failed ({psc}) for {email}: {getattr(pr,'text','')[:300]}")

            if args.rate_limit > 0:
                time.sleep(args.rate_limit)

        except requests.RequestException as e:
            logging.error(f"[{i}] Network error for {user.get('email','<no email>')}: {e}")
        except Exception as e:
            logging.error(f"[{i}] Unexpected error: {e}")

    logging.info(f"Done. {success}/{len(users_slice)} users posted preferences successfully.")

if __name__ == "__main__":
    main()
