# Clifford / Geometric Algebra Network

Full Clifford algebra (Cl(4,0)) for ETF feature representation. Each ETF is encoded as a multivector containing:
- scalar: last daily return
- vector: sensitivities to 4 macro factors
- bivector: top 6 pairwise correlations with other ETFs

The neural network uses Clifford‑equivariant linear layers and the geometric product. It is trained daily on a rolling 252‑day window to predict next‑day returns. Output is ranked by predicted return.

- **Run daily** via GitHub Actions
- **Results** stored on Hugging Face
- **Dashboard** shows top 3 ETFs and full ranking table

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
