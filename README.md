## Setup
Run
```bash
cp .env.example .env
```

```bash
uvicorn app.main:app --reload --port 9000
```

Then connect with
```bash
curl 'http://127.0.0.1:9000/api/booking/Q80chI4GLq4d4D3UR9p5/recommend'
```