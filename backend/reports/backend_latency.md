# Backend latency per input combination

Real pipeline via TestClient, 5 runs each, Tesseract OCR, live WHOIS/TLS checks disabled (add roughly the 3 s timeout per slow lookup when enabled; EasyOCR adds ~2-3 s per image on CPU).

| Inputs | Mean ms | Median ms | Max ms |
|---|---|---|---|
| text | 22 | 15 | 51 |
| image | 599 | 595 | 617 |
| url | 132 | 2 | 650 |
| text+image | 612 | 610 | 626 |
| text+url | 16 | 16 | 17 |
| image+url | 603 | 604 | 610 |
| text+image+url | 616 | 613 | 632 |
