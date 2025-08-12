# NBA Sports Muse

AI-powered NBA analytics dashboard that answers basketball questions with data visualizations and machine learning predictions.

## File Structure

| File | Description |
|------|-------------|
| `main.py` | Application entry point |
| `src/dashboard.py` | Streamlit web interface and user interaction |
| `src/query_router.py` | Routes queries to appropriate handlers |
| `src/query_handler.py` | Core query processing logic |
| `src/llm_processor.py` | Natural language query parsing |
| `src/nba_api_client.py` | NBA data fetching and caching |
| `src/ml_predictor.py` | Machine learning predictions for player stats |
| `src/visualizations.py` | Chart creation and data visualization |
| `src/pdf_exporter.py` | PDF report generation |
| `src/constants.py` | Application constants and configuration |

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run main.py
```

3. Open your browser to the URL shown (typically http://localhost:8501)

4. Ask NBA questions like:
   - "Compare Lebron to Curry"
   - "Predict Coby White's stats over the next 4 years"
   - "Warriors All time stats"

## Features

- Player comparisons with radar charts
- Multi-year statistical predictions
- Team analysis and franchise history
- PDF export of all reports
- Natural language query understanding