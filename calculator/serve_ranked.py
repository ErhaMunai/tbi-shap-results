"""SHAP-ranked, editable TBI research calculator with frozen Model A inference."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from risk_model import DEFAULT_BUNDLE, FEATURES, ModelA, validate_complete
from rankings import DEFAULT_RANKINGS, load_rankings


LOG = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
BODY_MAX = 4096
CONTENT_POLICY = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                  "connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
BACKGROUND_UNAVAILABLE = "The training reference is unavailable. Check the training imputation folder in the server terminal."
EXPLANATION_FAILED = "The probability was calculated, but the SHAP explanation did not complete. Check the server terminal."


def _error(message, status):
    return JSONResponse({"error": message}, status_code=status)


async def _read_body(request, site_origin):
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "application/json":
        return _error("Send the request as JSON.", 415)
    origin = request.headers.get("origin")
    expected = site_origin or f"http://{request.headers.get('host', '')}"
    if origin and origin != expected:
        return _error("Submit the request from the study calculator.", 403)
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > BODY_MAX:
            return _error("The request exceeds the size limit.", 413)
        chunks.extend(chunk)
    try:
        data = json.loads(chunks)
    except (ValueError, UnicodeError, RecursionError):
        return _error("Invalid request format.", 400)
    if not isinstance(data, dict) or set(data) != {*FEATURES, "eligible"}:
        return _error("Submit eligibility confirmation and all eight predictors.", 422)
    if data.pop("eligible") is not True:
        return _error("Confirm that the case meets the 24-hour cohort criteria.", 422)
    return data


def _estimate(model, explainer, data):
    if explainer is None:
        return model.predict_complete(data), None, "unavailable", BACKGROUND_UNAVAILABLE
    try:
        explanation = explainer.explain(data)
        return explanation["predicted_probability"], explanation, "ready", None
    except Exception as error:
        LOG.error("Local SHAP failed (%s). Check the training reference and model runtime.", type(error).__name__)
        return model.predict_complete(data), None, "failed", EXPLANATION_FAILED


def create_app(model=None, rankings=None, *, explainer=None, site_origin=None):
    if model is not None and rankings is not None:
        counts = {cohort["n_imputations"] for cohort in rankings["cohorts"]}
        if counts != {len(model.imputation_ids)}:
            raise ValueError("SHAP rankings and the frozen model use different imputation counts.")
    prediction_slot = asyncio.Semaphore(1)
    app = FastAPI(title="TBI Mortality Risk Calculator", docs_url=None, redoc_url=None, openapi_url=None)
    allowed_hosts = ["127.0.0.1", "localhost"]
    if site_origin:
        allowed_hosts.append(urlsplit(site_origin).hostname)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def page_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = CONTENT_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    async def status():
        return {"ready": model is not None, "model_id": model.model_id if model else None,
                "imputation_count": len(model.imputation_ids) if model else None,
                "ranking_ready": rankings is not None,
                "explanation_ready": explainer is not None}

    @app.get("/api/rankings")
    async def get_rankings():
        if rankings is None:
            return _error("SHAP rankings are unavailable. Check the server terminal.", 503)
        return rankings

    @app.post("/api/predict")
    async def predict(request: Request):
        data = await _read_body(request, site_origin)
        if isinstance(data, JSONResponse):
            return data
        if model is None:
            return _error("Frozen Model A is unavailable. Check the server terminal.", 503)
        if rankings is None:
            return _error("SHAP rankings are unavailable. Check the server terminal.", 503)
        try:
            checked = validate_complete(data)
        except ValueError as error:
            return _error(str(error), 422)
        try:
            async with prediction_slot:
                probability, explanation, state, message = await run_in_threadpool(_estimate, model, explainer, checked)
        except Exception:
            LOG.error("Prediction failed. Check the training environment and the frozen bundle.")
            return _error("The prediction could not be calculated. Check the server terminal.", 500)
        return {"probability": probability, "model_id": model.model_id,
                "predictor_count": len(FEATURES), "imputation_count": len(model.imputation_ids),
                "age_used": checked["age"], "explanation": explanation,
                "explanation_status": state, "explanation_message": message}

    return app


def https_origin(value):
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Enter the trusted proxy HTTPS origin.") from exc
    if (parsed.scheme != "https" or not parsed.hostname or "*" in parsed.netloc
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError("Enter the trusted proxy HTTPS origin, such as https://tbi.example.org.")
    return f"https://{parsed.netloc}"


def main():
    import uvicorn

    from local_shap import DEFAULT_MI_DIR, LocalShap

    parser = argparse.ArgumentParser(description="Run the TBI 24-hour mortality risk calculator.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="Trusted frozen model_bundle.joblib")
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS, help="SHAP results.json from the study")
    parser.add_argument("--mi-dir", type=Path, default=DEFAULT_MI_DIR, help="Original training imputation folder")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--site-origin", type=https_origin, help="Origin of the local institutional HTTPS proxy")
    args = parser.parse_args()
    try:
        rankings = load_rankings(args.rankings)
    except Exception as error:
        print(f"SHAP rankings unavailable: {type(error).__name__}: {error}", flush=True)
        rankings = None
    try:
        model = ModelA.load(args.bundle)
    except Exception as error:
        print(f"Frozen model unavailable: {type(error).__name__}: {error}", flush=True)
        model = None
    if model is not None and rankings is not None:
        if {cohort["n_imputations"] for cohort in rankings["cohorts"]} != {len(model.imputation_ids)}:
            print("The SHAP rankings and the frozen model must use matching imputation counts.", flush=True)
            rankings = None
    explainer = None
    if model is not None:
        print(f"Loaded {model.model_id} with {len(model.imputation_ids)} imputation models.", flush=True)
        if rankings is not None:
            try:
                explainer = LocalShap.load(model, args.mi_dir)
            except Exception as error:
                print(f"Patient-level SHAP unavailable: {type(error).__name__}: {error}", flush=True)
            else:
                print("Patient-level SHAP is ready.", flush=True)
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(model, rankings, explainer=explainer, site_origin=args.site_origin),
                host="127.0.0.1", port=args.port, access_log=False, proxy_headers=False)


if __name__ == "__main__":
    main()
