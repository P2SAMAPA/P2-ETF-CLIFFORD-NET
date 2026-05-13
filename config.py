import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-clifford-net-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Macro columns for vector part (grade-1)
MACRO_COLUMNS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]   # use 4 macro factors

# Rolling window for training (days)
TRAIN_WINDOW = 252

# Network hyperparameters
CLIFFORD_ALGEBRA = "Cl(4,0)"   # 4 basis vectors
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_MV_DIM = 16   # hidden multivector dimension (number of blades)

TOP_N = 3
