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
6. **Multi-Criteria Inventory Classification (Sales & POS Hits Methodology)**
   - **Step 1 — Separate Pareto Classification**: Ranks items independently on **Sales ($)** and **POS (Point-of-Sale) Hits** into Pareto classes:
     - Class A: 0% to 60% cumulative share
     - Class B: 61% to 80% cumulative share
     - Class C: 81% to 90% cumulative share
     - Class D: 91% to 100% cumulative share
     - Converted to numerical scores: $A = 4, B = 3, C = 2, D = 1$.
   - **Step 2 — Combined Weighted Score**: Weighted at 65% for Sales and 35% for POS Hits:
     $$\text{Combined Weighted Score} = 0.65(\text{Sales Score}) + 0.35(\text{Hits Score})$$
     *(e.g., Sales A [4] and POS Hits B [3] = $0.65(4) + 0.35(3) = 3.65$).*
   - **Step 3 — Final Inventory Class Assignment**:
     - Final Class A: Combined Weighted Score of **4.00 to 3.25**
     - Final Class B: Combined Weighted Score of **3.24 to 2.50**
     - Final Class C: Combined Weighted Score of **2.49 to 1.75**
     - Final Class D: Combined Weighted Score of **less than 1.75**
   - **Required Classification Summary**: Reports the number of items and summarizes total sales in each category (A, B, C, and D), alongside % of items, % of sales, POS hits, and 2D cross-classification matrices.

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

