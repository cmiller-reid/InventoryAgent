# Supply Chain Analytics Inventory Decision Agent

An intelligent, deterministic Supply Chain Decision Agent built for **BAN 517** to solve core inventory replenishment and optimization problems from natural language queries.

---

## 🏗️ 7-Stage Decision Architecture

The agent strictly follows the end-to-end analytical pipeline:

```
Customer Question
       │
       ▼
 1. Identify Model       (EOQ, Quantity Discount, Safety Stock & ROP, Fixed-Period, Newsvendor)
       │
       ▼
 2. Extract Inputs       (Regex + Semantic Parser, Unit Normalizer, Missing Parameter Detector)
       │
       ▼
 3. Select Python Tool   (Maps to deterministic functions in inventory_tools.py)
       │
       ▼
 4. Calculate            (Pure Python numerical calculations)
       │
       ▼
 5. Verify               (Mathematical consistency, operational bounds, edge-case checks)
       │
       ▼
 6. Explain              (Executive summary, KaTeX math derivation, cost trade-offs, managerial insights)
```

---

## 📦 Supported Inventory Models

1. **Economic Order Quantity (EOQ)**
   - Balances annual holding costs against ordering/setup costs.
   - Computes exact and practical integer $Q^*$, cycle days, order frequency, and annual cost breakdown.
2. **Quantity Discount Model (All-Units)**
   - Evaluates price-break schedules across multiple tiers.
   - Tests unconstrained $EOQ_i$ and feasibility at bracket thresholds to identify the global minimum total annual cost.
3. **Safety Stock & Reorder Point (Continuous Review $(s, Q)$)**
   - Handles three uncertainty scenarios:
     - Variable Demand, Constant Lead Time
     - Constant Demand, Variable Lead Time
     - Both Demand and Lead Time Variable
   - Maps Cycle Service Level ($CSL$) to exact standard normal $Z$-scores.
4. **Fixed-Period Review Model (Periodic Review $(P, S)$ / Order-Up-To)**
   - Computes target inventory level $M$ protecting over the protection interval $P + L$.
   - Determines replenishment quantity considering on-hand and pipeline inventory.
5. **Newsvendor Model (Single-Period Probabilistic)**
   - Determines critical fractile $CR = \frac{C_u}{C_u + C_o}$.
   - Computes optimal order quantity, expected lost sales, expected leftover inventory, and maximum expected net profit using standard normal loss functions.
6. **Multi-Criteria Inventory Classification (A / B / C / D Analysis)**
   - Performs dual-criterion Pareto stratification on uploaded CSV data using **Annual Sales ($)** and **Order Hits**.
   - Computes cumulative contribution percentages (A: $\le 60\%$, B: $60-80\%$, C: $80-90\%$, D: $> 90\%$).
   - Calculates weighted composite rank scores ($A=4, B=3, C=2, D=1$) with configurable or default $50/50$ weights.
   - Assigns final composite classes ($A \ge 3.4, B \ge 2.8, C \ge 2.1, D < 2.1$) and generates 2D cross-classification matrices ($4 \times 4$ crosstabs) and managerial inventory policies.

---

## 🎓 8-Part Pedagogical Student Verification Framework

Every deterministic analytical tool in `inventory_tools.py` returns an 8-part structured pedagogical payload so students and instructors can step through and audit the calculations:

1. **Inputs Extracted**: All raw parameters extracted from natural language or direct inputs.
2. **Variable Definitions**: Clear textbook symbol definitions (e.g., $D, S, H, CSL, Z, \sigma_{LT}, C_u, C_o$).
3. **Formula**: Theoretical standard mathematical formulas for the model.
4. **Formula with Substituted Values**: The exact mathematical expressions populated with actual problem numerical parameters.
5. **Intermediate Calculations**: All intermediate step-by-step values (e.g., variance terms, $Z$-scores, cost components, unit loss functions).
6. **Final Answer**: Recommended order quantities (exact & integer-rounded), reorder points, and total costs.
7. **Verification Checks**: Automated mathematical and operational validity checks (e.g., cost balance at EOQ, buffer adequacy, service level bounds).
8. **Assumptions**: Explicit operational assumptions underpinning the analytical model.

---

## 📂 Project Structure

| File | Role |
| :--- | :--- |
| `app.py` | Dual-mode frontend: Streamlit interactive dashboard with CSV data upload supporting multi-criteria classification (A/B/C/D) and inventory valuation, 1-click sample prompts, visual cost curves, dedicated 8-part student verification tabs, and step-by-step trace; plus interactive CLI mode. |
| `inventory_agent.py` | Orchestration agent executing the 7-stage decision pipeline, multi-pass semantic input extraction with synonym recognition, automated verification, and managerial explanation. |
| `inventory_tools.py` | Pure Python deterministic analytical calculation engine with dataclass outputs returning all 8 mandatory student verification sections, CSV validation, valuation analytics, and multi-criteria ABC/D inventory classification. |
| `tests/test_inventory_tools.py` | Pytest suite with 55 unit tests validating textbook calculations, input validation, synonym extraction, the 8-part student verification sections, CSV valuation calculations, and multi-criteria inventory classification. |
| `requirements.txt` | Dependency specifications (`streamlit`, `pandas`, `numpy`, `pytest`, `altair`). |

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web Dashboard
```bash
streamlit run app.py
```

### 3. Run via Command-Line Interface (CLI Mode)
```bash
python app.py
```

### 4. Run Unit Tests
```bash
python -m pytest tests/test_inventory_tools.py -v
```

