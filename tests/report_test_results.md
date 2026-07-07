# Technical Functionality Testing — Results

## Table 4.4  Functional Testing Results

| No | Function | Expected Result | Actual Result | Status |
|----|----------|-----------------|---------------|--------|
| 1 | Upload DICOM | Image uploaded successfully | HTTPSConnectionPool(host='pampered-enrage-girdle.ngrok-free. | Fail |
| 2 | Display Image | Image displayed in viewer | — | Fail |
| 3 | AI Prediction | Disease prediction generated | — | Fail |
| 4 | Report Generation | Report generated | — | Fail |
| 5 | Export PDF | PDF created | — | Fail |
| 6 | Save Study | Record stored in database | — | Fail |
| 7 | Search History | Study retrieved | — | Fail |
| 8 | Delete Study | Record removed | — | Fail |
| 9 | Settings Connection | API connected | HTTP 404 | Fail |

**0/9 functions passed.**

## 4.5.2.1  Stability Testing

- DICOM images processed : **10**
- Successful analyses    : **0**
- Crashes                : **10**
- System interruptions   : **0**

## Table 4.5  Error Handling Results

| Scenario | Expected Behaviour | Result |
|----------|--------------------|--------|
| Unsupported file | Rejected | Pass |
| API offline | Warning displayed | Pass |
| No file selected | Upload prevented | Pass |

**3/3 error scenarios handled correctly.**
