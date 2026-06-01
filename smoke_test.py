"""
smoke_test.py
Hits all API endpoints and prints PASS or FAIL.

Usage:
    python smoke_test.py https://your-hf-space-url.hf.space
    python smoke_test.py http://localhost:7860
"""

import sys
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"


def test_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=30)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        print("✅ GET /health        PASS")
        return True
    except Exception as e:
        print(f"❌ GET /health        FAIL — {e}")
        return False


def test_predict():
    try:
        # Create dummy image
        import io
        from PIL import Image
        import numpy as np

        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        r = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "predicted_class" in data
        assert "probability" in data
        assert "risk_level" in data
        print("✅ POST /predict      PASS")
        return True
    except Exception as e:
        print(f"❌ POST /predict      FAIL — {e}")
        return False


def test_history():
    try:
        r = requests.get(f"{BASE_URL}/history", timeout=30)
        assert r.status_code == 200
        assert "predictions" in r.json()
        print("✅ GET /history       PASS")
        return True
    except Exception as e:
        print(f"❌ GET /history       FAIL — {e}")
        return False


def test_stats():
    try:
        r = requests.get(f"{BASE_URL}/stats", timeout=30)
        assert r.status_code == 200
        print("✅ GET /stats         PASS")
        return True
    except Exception as e:
        print(f"❌ GET /stats         FAIL — {e}")
        return False


if __name__ == "__main__":
    print("=" * 45)
    print(f"  MediScan Smoke Test")
    print(f"  URL: {BASE_URL}")
    print("=" * 45)

    results = [
        test_health(),
        test_predict(),
        test_history(),
        test_stats(),
    ]

    print("=" * 45)
    if all(results):
        print("  ✅ ALL TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("=" * 45)