"""
inventory_tools.py
==================
Core deterministic analytical calculation engine for Supply Chain Inventory Decision Agent.

Implements deterministic analytical tools for:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (Safety Stock and Reorder Point)
4. calculate_periodic_review() (Fixed-Period Review / Order-Up-To)
5. calculate_newsvendor() (Single-Period Newsvendor Model)

Every calculation tool returns the complete 8-part pedagogical verification structure:
1. inputs_extracted (or inputs)
2. variable_definitions
3. formula
4. formula_with_substituted_values
5. intermediate_calculations
6. final_answer
7. verification_checks
8. assumptions

All calculations are strictly deterministic Python numerical calculations.
No missing numerical inputs are invented.
"""

import math
import statistics
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# Standard Normal Distribution Utilities
# =============================================================================

def standard_normal_pdf(z: float) -> float:
    """Standard normal probability density function: phi(z)."""
    return statistics.NormalDist(0.0, 1.0).pdf(z)


def standard_normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function: Phi(z)."""
    return statistics.NormalDist(0.0, 1.0).cdf(z)


def standard_normal_inv_cdf(p: float) -> float:
    """Standard normal inverse CDF / quantile function: Phi^{-1}(p)."""
    if p is None:
        raise ValueError("Service level or probability cannot be None.")
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Service level or probability must be strictly between 0 and 1, got {p}")
    return statistics.NormalDist(0.0, 1.0).inv_cdf(p)


def standard_normal_loss(z: float) -> float:
    """
    Standard normal unit loss function: L(z) = phi(z) - z * (1 - Phi(z)).
    Evaluates expected stockout / lost sales for standard normal distribution.
    """
    pdf_z = standard_normal_pdf(z)
    cdf_z = standard_normal_cdf(z)
    return pdf_z - z * (1.0 - cdf_z)


# =============================================================================
# 1. Economic Order Quantity (EOQ)
# =============================================================================

@dataclass
class EOQResult:
    model: str
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, str]]
    assumptions: List[str]

    # Direct attribute access for backwards-compatibility & convenient usage
    demand: float
    ordering_cost: float
    holding_cost: float
    unit_cost: Optional[float]
    work_days_per_year: int
    eoq: float
    eoq_rounded: int
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_inventory_cost: float
    annual_purchase_cost: Optional[float]
    total_annual_cost: Optional[float]
    orders_per_year: float
    cycle_time_years: float
    cycle_time_days: float
    daily_demand: float

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_eoq(
    demand: float,
    ordering_cost: float,
    holding_cost: float,
    unit_cost: Optional[float] = None,
    work_days_per_year: int = 250
) -> EOQResult:
    """
    Calculates the Economic Order Quantity (EOQ) and comprehensive inventory cost metrics.

    Formula:
        Q = sqrt((2 * D * S) / H)

    Returns the complete 8-part verification breakdown:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    # 1. Validate inputs (Never invent missing numerical inputs)
    if demand is None:
        raise ValueError("Annual demand (D) is required and cannot be None.")
    if ordering_cost is None:
        raise ValueError("Ordering cost (S) is required and cannot be None.")
    if holding_cost is None:
        raise ValueError("Holding cost (H) is required and cannot be None.")
    if work_days_per_year is None:
        raise ValueError("Work days per year cannot be None.")

    if not isinstance(demand, (int, float)) or demand <= 0:
        raise ValueError(f"Annual demand must be greater than 0, got {demand}")
    if not isinstance(ordering_cost, (int, float)) or ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be greater than 0, got {ordering_cost}")
    if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
        raise ValueError(f"Holding cost must be greater than 0, got {holding_cost}")
    if unit_cost is not None:
        if not isinstance(unit_cost, (int, float)) or unit_cost < 0:
            raise ValueError(f"Unit cost cannot be negative, got {unit_cost}")
    if not isinstance(work_days_per_year, int) or work_days_per_year <= 0:
        raise ValueError(f"Work days per year must be a positive integer, got {work_days_per_year}")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "holding_cost": holding_cost,
        "unit_cost": unit_cost,
        "work_days_per_year": work_days_per_year
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Ordering or setup cost per order ($/order)",
        "H": "Holding or carrying cost per unit per year ($/unit/year)",
        "C": "Unit purchase price ($/unit, optional)",
        "W": "Operating work days per year (days/year)",
        "Q": "Order batch quantity (units)",
        "EOQ": "Economic Order Quantity minimizing total inventory holding and setup costs (units)"
    }

    # Section 3: Standard Algebraic Formulas
    formula = {
        "eoq": "EOQ = sqrt((2 * D * S) / H)",
        "annual_ordering_cost": "Annual Ordering Cost = (D / Q) * S",
        "annual_holding_cost": "Annual Holding Cost = (Q / 2) * H",
        "total_annual_inventory_cost": "Total Inventory Cost = Annual Ordering Cost + Annual Holding Cost = (D/Q)*S + (Q/2)*H",
        "orders_per_year": "Orders Per Year = D / Q",
        "cycle_time_days": "Cycle Time (Days) = (Q / D) * W"
    }

    # 2. Intermediate numerical calculations
    eoq = math.sqrt((2.0 * demand * ordering_cost) / holding_cost)
    eoq_rounded = max(1, round(eoq))

    annual_ordering_cost = (demand / eoq) * ordering_cost
    annual_holding_cost = (eoq / 2.0) * holding_cost
    total_inventory_cost = annual_ordering_cost + annual_holding_cost

    annual_purchase_cost = (demand * unit_cost) if unit_cost is not None else None
    total_annual_cost = (total_inventory_cost + annual_purchase_cost) if annual_purchase_cost is not None else None

    orders_per_year = demand / eoq
    cycle_time_years = eoq / demand
    cycle_time_days = cycle_time_years * work_days_per_year
    daily_demand = demand / work_days_per_year

    # Section 4: Formulas with Substituted Values
    formula_with_substituted_values = {
        "eoq": (
            f"EOQ = sqrt((2 * {demand:,.2f} * {ordering_cost:,.2f}) / {holding_cost:,.2f}) "
            f"= sqrt({2.0 * demand * ordering_cost:,.2f} / {holding_cost:,.2f}) "
            f"= sqrt({(2.0 * demand * ordering_cost) / holding_cost:,.2f}) "
            f"= {eoq:,.2f} units (Practical batch: {eoq_rounded:,} units)"
        ),
        "annual_ordering_cost": (
            f"Annual Ordering Cost = ({demand:,.2f} / {eoq:,.2f}) * ${ordering_cost:,.2f} "
            f"= ${annual_ordering_cost:,.2f}"
        ),
        "annual_holding_cost": (
            f"Annual Holding Cost = ({eoq:,.2f} / 2) * ${holding_cost:,.2f} "
            f"= ${annual_holding_cost:,.2f}"
        ),
        "total_annual_inventory_cost": (
            f"Total Inventory Cost = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} "
            f"= ${total_inventory_cost:,.2f}"
        ),
        "orders_per_year": (
            f"Orders Per Year = {demand:,.2f} / {eoq:,.2f} = {orders_per_year:.2f} orders/year"
        ),
        "cycle_time_days": (
            f"Cycle Time = ({eoq:,.2f} / {demand:,.2f}) * {work_days_per_year} "
            f"= {cycle_time_days:.1f} operating days"
        )
    }

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "numerator_2DS": round(2.0 * demand * ordering_cost, 2),
        "denominator_H": round(holding_cost, 4),
        "ratio_2DS_over_H": round((2.0 * demand * ordering_cost) / holding_cost, 4),
        "eoq_exact": round(eoq, 4),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_annual_inventory_cost": round(total_inventory_cost, 2),
        "annual_purchase_cost": round(annual_purchase_cost, 2) if annual_purchase_cost is not None else None,
        "total_annual_cost": round(total_annual_cost, 2) if total_annual_cost is not None else None,
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_days": round(cycle_time_days, 1),
        "daily_demand": round(daily_demand, 2)
    }

    # Section 6: Final Answer
    final_answer = {
        "optimal_order_quantity": round(eoq, 2),
        "optimal_order_quantity_exact": round(eoq, 2),
        "recommended_order_quantity": eoq_rounded,
        "minimum_annual_inventory_cost": round(total_inventory_cost, 2),
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_days": round(cycle_time_days, 1),
        "total_annual_cost": round(total_annual_cost, 2) if total_annual_cost is not None else None
    }

    # Section 7: Deterministic Verification Checks
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    is_cost_balanced = cost_diff <= 1.0 or (annual_ordering_cost > 0 and (cost_diff / annual_ordering_cost) < 0.01)

    verification_checks = [
        {
            "check_name": "Cost Balance Optimality",
            "condition": "Annual Ordering Cost == Annual Holding Cost at exact EOQ",
            "status": "PASS" if is_cost_balanced else "WARNING",
            "details": f"Annual ordering cost (${annual_ordering_cost:,.2f}) equals annual holding cost (${annual_holding_cost:,.2f}) at minimum total cost."
        },
        {
            "check_name": "Non-Zero Positive Parameters",
            "condition": "D > 0, S > 0, H > 0",
            "status": "PASS",
            "details": f"Demand ({demand:,.0f}), ordering cost (${ordering_cost:,.2f}), and holding cost (${holding_cost:,.2f}) are strictly positive."
        },
        {
            "check_name": "Order Cycle Feasibility",
            "condition": "Cycle time <= Operating work days per year",
            "status": "PASS" if cycle_time_days <= work_days_per_year else "WARNING",
            "details": f"Order cycle time is {cycle_time_days:.1f} days ({orders_per_year:.2f} orders/year)."
        }
    ]

    # Section 8: Model Assumptions
    assumptions = [
        "Demand rate is constant, deterministic, and continuous over time.",
        "Replenishment lead time is zero or constant and known.",
        "Replenishment arrives in a single batch instantaneously.",
        "Holding cost per unit per year is constant and linear with average inventory (Q/2).",
        "Fixed ordering/setup cost is incurred per order, independent of order quantity.",
        "No stockouts, shortages, or backorders are permitted.",
        "Unit purchase price is constant with no volume discounts."
    ]

    return EOQResult(
        model="Economic Order Quantity (EOQ)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        demand=demand,
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
        unit_cost=unit_cost,
        work_days_per_year=work_days_per_year,
        eoq=round(eoq, 2),
        eoq_rounded=eoq_rounded,
        annual_ordering_cost=round(annual_ordering_cost, 2),
        annual_holding_cost=round(annual_holding_cost, 2),
        total_annual_inventory_cost=round(total_inventory_cost, 2),
        annual_purchase_cost=round(annual_purchase_cost, 2) if annual_purchase_cost is not None else None,
        total_annual_cost=round(total_annual_cost, 2) if total_annual_cost is not None else None,
        orders_per_year=round(orders_per_year, 2),
        cycle_time_years=round(cycle_time_years, 4),
        cycle_time_days=round(cycle_time_days, 1),
        daily_demand=round(daily_demand, 2)
    )


def generate_eoq_cost_curve(
    demand: float,
    ordering_cost: float,
    holding_cost: float,
    eoq: float,
    points: int = 50
) -> List[Dict[str, float]]:
    """Generates cost curve data for visualization around EOQ."""
    min_q = max(1.0, eoq * 0.2)
    max_q = eoq * 2.2
    step = (max_q - min_q) / (points - 1)

    curve = []
    for i in range(points):
        q = min_q + i * step
        order_cost = (demand / q) * ordering_cost
        hold_cost = (q / 2.0) * holding_cost
        tot_cost = order_cost + hold_cost
        curve.append({
            "order_quantity": round(q, 1),
            "ordering_cost": round(order_cost, 2),
            "holding_cost": round(hold_cost, 2),
            "total_inventory_cost": round(tot_cost, 2)
        })
    return curve


# =============================================================================
# 2. Quantity Discount Model
# =============================================================================

@dataclass
class DiscountTierEvaluation:
    tier_index: int
    min_quantity: float
    max_quantity: Optional[float]
    unit_price: float
    holding_cost: float
    raw_eoq: float
    is_raw_eoq_feasible: bool
    candidate_quantity: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountResult:
    model: str
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, str]]
    assumptions: List[str]

    # Direct attribute access
    demand: float
    ordering_cost: float
    holding_cost_rate: Optional[float]
    constant_holding_cost: Optional[float]
    tier_evaluations: List[DiscountTierEvaluation]
    optimal_tier_index: int
    optimal_order_quantity: float
    optimal_unit_price: float
    optimal_total_annual_cost: float
    baseline_total_annual_cost: Optional[float]
    total_savings: Optional[float]
    savings_percentage: Optional[float]

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_quantity_discount(
    demand: float,
    ordering_cost: float,
    discount_brackets: List[Dict[str, Any]],
    holding_cost_rate: Optional[float] = None,
    constant_holding_cost: Optional[float] = None
) -> QuantityDiscountResult:
    """
    Evaluates feasible order quantities and compares total annual cost across discount tiers.
    Supports fixed holding cost ($/unit/year) and percentage holding cost rate (% of unit price).

    Returns the complete 8-part verification breakdown:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    # 1. Validate inputs (Never invent missing numerical inputs)
    if demand is None:
        raise ValueError("Annual demand (D) is required and cannot be None.")
    if ordering_cost is None:
        raise ValueError("Ordering cost (S) is required and cannot be None.")
    if discount_brackets is None or not discount_brackets:
        raise ValueError("discount_brackets cannot be None or empty.")

    if not isinstance(demand, (int, float)) or demand <= 0:
        raise ValueError(f"Annual demand must be > 0, got {demand}")
    if not isinstance(ordering_cost, (int, float)) or ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be > 0, got {ordering_cost}")

    if holding_cost_rate is None and constant_holding_cost is None:
        raise ValueError("Must provide either holding_cost_rate (e.g., 0.20) or constant_holding_cost.")
    if holding_cost_rate is not None and (not isinstance(holding_cost_rate, (int, float)) or holding_cost_rate <= 0):
        raise ValueError(f"Holding cost rate must be > 0, got {holding_cost_rate}")
    if constant_holding_cost is not None and (not isinstance(constant_holding_cost, (int, float)) or constant_holding_cost <= 0):
        raise ValueError(f"Constant holding cost must be > 0, got {constant_holding_cost}")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "holding_cost_rate": holding_cost_rate,
        "constant_holding_cost": constant_holding_cost,
        "discount_brackets": discount_brackets
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "D": "Annual customer demand (units/year)",
        "S": "Ordering cost per order ($/order)",
        "C_i": "Unit purchase price for discount tier i ($/unit)",
        "h": "Annual holding cost rate (% of purchase price)",
        "H_i": "Holding cost for tier i ($/unit/year) = h * C_i or constant H",
        "EOQ_i": "Unconstrained economic order quantity for tier i = sqrt((2 * D * S) / H_i)",
        "Q_cand_i": "Feasible candidate order quantity for tier i (units)",
        "TC_i": "Total annual cost for tier i = Purchase Cost + Ordering Cost + Holding Cost"
    }

    # Section 3: Standard Algebraic Formulas
    formula = {
        "tier_holding_cost": "H_i = (h * C_i) if percentage_rate else constant_H",
        "raw_eoq": "EOQ_i = sqrt((2 * D * S) / H_i)",
        "feasibility_rule": "Candidate Q_i = EOQ_i if min_Q <= EOQ_i <= max_Q; else min_Q if EOQ_i < min_Q; else max_Q",
        "annual_purchase_cost": "Purchase Cost_i = D * C_i",
        "annual_ordering_cost": "Ordering Cost_i = (D / Q_cand_i) * S",
        "annual_holding_cost": "Holding Cost_i = (Q_cand_i / 2) * H_i",
        "total_annual_cost": "TC_i = Purchase Cost_i + Ordering Cost_i + Holding Cost_i",
        "optimal_policy": "Optimal Policy = min(TC_i across all feasible candidates)"
    }

    sorted_brackets = sorted(discount_brackets, key=lambda x: x["min_qty"])
    evaluations: List[DiscountTierEvaluation] = []
    substituted_tier_details: List[str] = []

    for idx, bracket in enumerate(sorted_brackets):
        if "min_qty" not in bracket or "unit_price" not in bracket:
            raise ValueError(f"Each tier must contain 'min_qty' and 'unit_price', invalid tier: {bracket}")
        min_q = float(bracket["min_qty"])
        max_q = float(bracket["max_qty"]) if bracket.get("max_qty") is not None else None
        price = float(bracket["unit_price"])

        if min_q < 0:
            raise ValueError(f"Tier min_qty cannot be negative, got {min_q}")
        if price <= 0:
            raise ValueError(f"Unit price must be > 0, got {price} in tier {idx + 1}")
        if max_q is not None and max_q < min_q:
            raise ValueError(f"Tier max_qty ({max_q}) cannot be less than min_qty ({min_q})")

        # Determine tier holding cost
        if holding_cost_rate is not None:
            tier_h = holding_cost_rate * price
        else:
            tier_h = constant_holding_cost

        # Unconstrained EOQ for this tier
        raw_eoq = math.sqrt((2.0 * demand * ordering_cost) / tier_h)
        is_feasible = (raw_eoq >= min_q) and (max_q is None or raw_eoq <= max_q)

        if is_feasible:
            cand_q = raw_eoq
        elif raw_eoq < min_q:
            cand_q = min_q
        else:
            cand_q = max_q

        # Annual costs for candidate quantity
        purchase_cost = demand * price
        order_cost = (demand / cand_q) * ordering_cost
        hold_cost = (cand_q / 2.0) * tier_h
        tot_cost = purchase_cost + order_cost + hold_cost

        evaluations.append(DiscountTierEvaluation(
            tier_index=idx + 1,
            min_quantity=min_q,
            max_quantity=max_q,
            unit_price=price,
            holding_cost=round(tier_h, 2),
            raw_eoq=round(raw_eoq, 2),
            is_raw_eoq_feasible=is_feasible,
            candidate_quantity=round(cand_q, 2),
            annual_purchase_cost=round(purchase_cost, 2),
            annual_ordering_cost=round(order_cost, 2),
            annual_holding_cost=round(hold_cost, 2),
            total_annual_cost=round(tot_cost, 2)
        ))

        max_label = f"{max_q:,.0f}" if max_q is not None else "inf"
        substituted_tier_details.append(
            f"Tier {idx + 1} [{min_q:,.0f} - {max_label} units @ ${price:.2f}]: "
            f"H_{idx+1} = ${tier_h:.2f}, Raw EOQ = sqrt((2 * {demand:,.0f} * {ordering_cost:,.2f}) / {tier_h:.2f}) = {raw_eoq:.2f} "
            f"({'Feasible' if is_feasible else f'Adjusted to Cutoff {cand_q:,.0f}'}) -> "
            f"TC = ({demand:,.0f} * ${price:.2f}) + (({demand:,.0f} / {cand_q:.0f}) * ${ordering_cost:.2f}) + (({cand_q:.0f} / 2) * ${tier_h:.2f}) "
            f"= ${purchase_cost:,.2f} + ${order_cost:,.2f} + ${hold_cost:,.2f} = ${tot_cost:,.2f}"
        )

    # Pick tier with minimum total annual cost
    best_eval = min(evaluations, key=lambda x: x.total_annual_cost)
    baseline_eval = evaluations[0]
    total_savings = baseline_eval.total_annual_cost - best_eval.total_annual_cost
    savings_pct = (total_savings / baseline_eval.total_annual_cost * 100.0) if baseline_eval.total_annual_cost > 0 else 0.0

    # Section 4: Formulas with Substituted Values
    formula_with_substituted_values = {
        "tier_step_by_step_substitutions": "\n".join(substituted_tier_details),
        "optimal_selection_summary": (
            f"Minimizing total cost across all candidate tiers identifies Tier {best_eval.tier_index} "
            f"with candidate order quantity Q* = {best_eval.candidate_quantity:,.0f} units at ${best_eval.unit_price:.2f}/unit, "
            f"achieving lowest Total Annual Cost = ${best_eval.total_annual_cost:,.2f} "
            f"(Annual Savings vs Base Tier: ${total_savings:,.2f} or {savings_pct:.1f}%)."
        )
    }

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "tier_evaluations": [asdict(e) for e in evaluations],
        "baseline_total_annual_cost": baseline_eval.total_annual_cost,
        "holding_cost_type": "percentage_of_price" if holding_cost_rate is not None else "fixed_dollar",
        "number_of_tiers_evaluated": len(evaluations)
    }

    # Section 6: Final Answer
    final_answer = {
        "optimal_tier_index": best_eval.tier_index,
        "optimal_order_quantity": best_eval.candidate_quantity,
        "optimal_unit_price": best_eval.unit_price,
        "optimal_total_annual_cost": best_eval.total_annual_cost,
        "annual_purchase_cost": best_eval.annual_purchase_cost,
        "annual_ordering_cost": best_eval.annual_ordering_cost,
        "annual_holding_cost": best_eval.annual_holding_cost,
        "annual_savings_vs_baseline": round(max(0.0, total_savings), 2),
        "savings_percentage": round(max(0.0, savings_pct), 2)
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check_name": "Global Cost Minimization",
            "condition": "Optimal tier TC <= TC of all other tiers",
            "status": "PASS",
            "details": f"Tier {best_eval.tier_index} confirmed minimum annual expenditure at ${best_eval.total_annual_cost:,.2f}."
        },
        {
            "check_name": "Threshold Qualification",
            "condition": "Candidate quantity qualifies for tier discount price",
            "status": "PASS" if best_eval.candidate_quantity >= best_eval.min_quantity else "FAIL",
            "details": f"Order quantity {best_eval.candidate_quantity:,.0f} qualifies for Tier {best_eval.tier_index} (min required: {best_eval.min_quantity:,.0f})."
        },
        {
            "check_name": "Holding Cost Positivity",
            "condition": "Holding cost rate or fixed holding cost > 0",
            "status": "PASS",
            "details": f"Holding cost evaluated as {'percentage' if holding_cost_rate is not None else 'fixed'} (${best_eval.holding_cost:.2f}/unit/year)."
        }
    ]

    # Section 8: Assumptions
    assumptions = [
        "All-units quantity discount schedule applies to entire batch size ordered.",
        "Demand is deterministic and constant over the annual planning horizon.",
        "No shortages, stockouts, or backorders allowed.",
        "Inventory carrying cost is based on fixed dollar amount per unit or constant percentage of unit price.",
        "Replenishment lead time is instantaneous or constant and known."
    ]

    return QuantityDiscountResult(
        model="Quantity Discount Model",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        demand=demand,
        ordering_cost=ordering_cost,
        holding_cost_rate=holding_cost_rate,
        constant_holding_cost=constant_holding_cost,
        tier_evaluations=evaluations,
        optimal_tier_index=best_eval.tier_index,
        optimal_order_quantity=best_eval.candidate_quantity,
        optimal_unit_price=best_eval.unit_price,
        optimal_total_annual_cost=best_eval.total_annual_cost,
        baseline_total_annual_cost=baseline_eval.total_annual_cost,
        total_savings=round(max(0.0, total_savings), 2),
        savings_percentage=round(max(0.0, savings_pct), 2)
    )


# =============================================================================
# 3. Safety Stock and Reorder Point (Continuous Review)
# =============================================================================

@dataclass
class SafetyStockROPResult:
    model: str
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, str]]
    assumptions: List[str]

    # Direct attribute access
    scenario: str
    lead_time: float
    avg_demand_per_period: float
    std_demand_per_period: float
    std_lead_time: float
    service_level: float
    z_score: float
    expected_lead_time_demand: float
    std_lead_time_demand: float
    sigma_LT: float
    safety_stock: float
    safety_stock_rounded: int
    reorder_point: float
    reorder_point_rounded: int
    stockout_risk_per_cycle: float
    annual_safety_stock_holding_cost: Optional[float]

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_rop(
    lead_time: float,
    avg_demand: float,
    std_demand: float = 0.0,
    std_lead_time: float = 0.0,
    service_level: float = 0.95,
    holding_cost: Optional[float] = None
) -> SafetyStockROPResult:
    """
    Calculates Safety Stock (SS) and Reorder Point (ROP) under continuous review (s, Q).
    Supports all four variance scenarios.

    Returns the complete 8-part verification breakdown:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    # 1. Validate inputs (Never invent missing numerical inputs)
    if lead_time is None:
        raise ValueError("Lead time (L) is required and cannot be None.")
    if avg_demand is None:
        raise ValueError("Average demand (d) is required and cannot be None.")
    if service_level is None:
        raise ValueError("Service level (CSL) is required and cannot be None.")

    if not isinstance(lead_time, (int, float)) or lead_time <= 0:
        raise ValueError(f"Lead time must be > 0, got {lead_time}")
    if not isinstance(avg_demand, (int, float)) or avg_demand <= 0:
        raise ValueError(f"Average demand must be > 0, got {avg_demand}")
    if not isinstance(std_demand, (int, float)) or std_demand < 0:
        raise ValueError(f"Demand std dev cannot be negative, got {std_demand}")
    if not isinstance(std_lead_time, (int, float)) or std_lead_time < 0:
        raise ValueError(f"Lead time std dev cannot be negative, got {std_lead_time}")
    if not isinstance(service_level, (int, float)) or not (0.0 < service_level < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 exclusive (e.g. 0.95), got {service_level}")
    if holding_cost is not None and (not isinstance(holding_cost, (int, float)) or holding_cost < 0):
        raise ValueError(f"Holding cost cannot be negative, got {holding_cost}")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "lead_time": lead_time,
        "avg_demand": avg_demand,
        "std_demand": std_demand,
        "std_lead_time": std_lead_time,
        "service_level": service_level,
        "holding_cost": holding_cost
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "L": "Average replenishment lead time (periods, e.g. days or weeks)",
        "d": "Average customer demand per period (units/period)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "sigma_L": "Standard deviation of replenishment lead time (periods)",
        "CSL": "Cycle Service Level / probability of not stocking out in a replenishment cycle",
        "z": "Standard normal safety factor = Phi^{-1}(CSL)",
        "sigma_LT": "Standard deviation of total demand during lead time (units)",
        "SS": "Safety stock buffer = z * sigma_LT (units)",
        "ROP": "Reorder point = Expected Lead Time Demand + SS = (d * L) + SS (units)"
    }

    # 2. Intermediate calculations & scenario detection
    z = standard_normal_inv_cdf(service_level)
    expected_ltd = avg_demand * lead_time

    if std_demand == 0.0 and std_lead_time == 0.0:
        scenario = "Constant Demand and Constant Lead Time"
        sigma_lt = 0.0
        sigma_formula = "sigma_LT = 0"
        sigma_sub = "sigma_LT = 0.0 units"
    elif std_demand > 0.0 and std_lead_time == 0.0:
        scenario = "Variable Demand and Constant Lead Time"
        sigma_lt = math.sqrt(lead_time) * std_demand
        sigma_formula = "sigma_LT = sqrt(L) * sigma_d"
        sigma_sub = (
            f"sigma_LT = sqrt({lead_time}) * {std_demand} "
            f"= {math.sqrt(lead_time):.4f} * {std_demand} = {sigma_lt:.2f} units"
        )
    elif std_demand == 0.0 and std_lead_time > 0.0:
        scenario = "Constant Demand and Variable Lead Time"
        sigma_lt = avg_demand * std_lead_time
        sigma_formula = "sigma_LT = d * sigma_L"
        sigma_sub = (
            f"sigma_LT = {avg_demand} * {std_lead_time} = {sigma_lt:.2f} units"
        )
    else:
        scenario = "Variable Demand and Variable Lead Time"
        var_lt = (lead_time * (std_demand ** 2)) + ((avg_demand ** 2) * (std_lead_time ** 2))
        sigma_lt = math.sqrt(var_lt)
        sigma_formula = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        sigma_sub = (
            f"sigma_LT = sqrt(({lead_time} * {std_demand}^2) + ({avg_demand}^2 * {std_lead_time}^2)) "
            f"= sqrt({lead_time * (std_demand**2):.2f} + {(avg_demand**2) * (std_lead_time**2):.2f}) "
            f"= sqrt({var_lt:.2f}) = {sigma_lt:.2f} units"
        )

    safety_stock = z * sigma_lt
    safety_stock_rounded = max(0, math.ceil(safety_stock))
    rop = expected_ltd + safety_stock
    rop_rounded = max(0, math.ceil(rop))

    stockout_risk = 1.0 - service_level
    annual_ss_cost = (safety_stock * holding_cost) if holding_cost is not None else None

    # Section 3: Standard Algebraic Formulas
    formula = {
        "expected_lead_time_demand": "Expected LTD = d * L",
        "sigma_LT": sigma_formula,
        "z_score": "z = Phi^{-1}(CSL)",
        "safety_stock": "SS = z * sigma_LT",
        "reorder_point": "ROP = Expected LTD + SS = (d * L) + SS"
    }

    # Section 4: Formulas with Substituted Values
    formula_with_substituted_values = {
        "expected_lead_time_demand": f"Expected LTD = {avg_demand} * {lead_time} = {expected_ltd:.2f} units",
        "sigma_LT": sigma_sub,
        "z_score": f"z = Phi^{{-1}}({service_level}) = {z:.4f}",
        "safety_stock": f"SS = {z:.4f} * {sigma_lt:.2f} = {safety_stock:.2f} units (Rounded: {safety_stock_rounded:,} units)",
        "reorder_point": f"ROP = {expected_ltd:.2f} + {safety_stock:.2f} = {rop:.2f} units (Rounded: {rop_rounded:,} units)"
    }

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "scenario": scenario,
        "expected_lead_time_demand": round(expected_ltd, 2),
        "sigma_LT": round(sigma_lt, 2),
        "z_score": round(z, 4),
        "safety_stock_exact": round(safety_stock, 2),
        "reorder_point_exact": round(rop, 2),
        "stockout_risk_per_cycle": round(stockout_risk, 4),
        "annual_safety_stock_holding_cost": round(annual_ss_cost, 2) if annual_ss_cost is not None else None
    }

    # Section 6: Final Answer
    final_answer = {
        "scenario": scenario,
        "reorder_point": rop_rounded,
        "reorder_point_exact": round(rop, 2),
        "safety_stock": safety_stock_rounded,
        "safety_stock_exact": round(safety_stock, 2),
        "expected_lead_time_demand": round(expected_ltd, 2),
        "cycle_service_level": service_level,
        "z_score": round(z, 4),
        "stockout_risk_per_cycle": round(stockout_risk, 4)
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check_name": "Buffer Protection Check",
            "condition": "ROP >= Expected Lead Time Demand",
            "status": "PASS" if rop >= expected_ltd else "FAIL",
            "details": f"Reorder point ({rop:.2f}) covers expected lead time demand ({expected_ltd:.2f}) plus {safety_stock:.2f} units safety stock."
        },
        {
            "check_name": "Cycle Service Level Validity",
            "condition": "0 < CSL < 1.0",
            "status": "PASS",
            "details": f"Cycle service level {service_level:.1%} maps to standard normal quantile Z = {z:.4f}."
        },
        {
            "check_name": "Safety Stock Non-negativity",
            "condition": "Safety Stock >= 0",
            "status": "PASS" if safety_stock >= 0 else "FAIL",
            "details": f"Safety stock ({safety_stock:.2f} units) is non-negative and properly buffered."
        }
    ]

    # Section 8: Model Assumptions
    assumptions = [
        "Demand across successive periods is independent and identically distributed.",
        "Lead time duration is independent of customer demand.",
        "Demand during lead time follows a normal distribution.",
        "Continuous inventory monitoring policy (s, Q) is in place.",
        "Unmet customer demand during lead time stockouts is fully backordered.",
        f"Cycle Service Level (CSL) target of {service_level:.1%} specifies the cycle in-stock probability."
    ]

    return SafetyStockROPResult(
        model="Safety Stock and Reorder Point (Continuous Review)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        scenario=scenario,
        lead_time=lead_time,
        avg_demand_per_period=avg_demand,
        std_demand_per_period=std_demand,
        std_lead_time=std_lead_time,
        service_level=service_level,
        z_score=round(z, 4),
        expected_lead_time_demand=round(expected_ltd, 2),
        std_lead_time_demand=round(sigma_lt, 2),
        sigma_LT=round(sigma_lt, 2),
        safety_stock=round(safety_stock, 2),
        safety_stock_rounded=safety_stock_rounded,
        reorder_point=round(rop, 2),
        reorder_point_rounded=rop_rounded,
        stockout_risk_per_cycle=round(stockout_risk, 4),
        annual_safety_stock_holding_cost=round(annual_ss_cost, 2) if annual_ss_cost is not None else None
    )


# Backward compatibility alias
calculate_safety_stock_rop = calculate_rop


# =============================================================================
# 4. Fixed-Period Review Model (Periodic Review / Order-Up-To)
# =============================================================================

@dataclass
class PeriodicReviewResult:
    model: str
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, str]]
    assumptions: List[str]

    # Direct attribute access
    review_period: float
    lead_time: float
    protection_period: float
    protection_interval: float  # alias
    avg_demand_per_period: float
    std_demand_per_period: float
    service_level: float
    z_score: float
    expected_demand_protection_interval: float
    std_demand_protection_interval: float
    safety_stock: float
    safety_stock_rounded: int
    target_inventory_level: float
    target_inventory_level_rounded: int
    current_inventory_position: float
    recommended_order_quantity: float
    recommended_order_quantity_rounded: int

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_periodic_review(
    review_period: float,
    lead_time: float,
    avg_demand: float,
    std_demand: float,
    service_level: float = 0.95,
    current_inventory: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0
) -> PeriodicReviewResult:
    """
    Calculates Target Inventory Level (Order-Up-To level S or M) and Order Quantity (Q)
    for a Fixed-Period Periodic Review system.

    Returns the complete 8-part verification breakdown:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    # 1. Validate inputs (Never invent missing numerical inputs)
    if review_period is None:
        raise ValueError("Review period (P) is required and cannot be None.")
    if lead_time is None:
        raise ValueError("Lead time (L) is required and cannot be None.")
    if avg_demand is None:
        raise ValueError("Average demand (d) is required and cannot be None.")
    if std_demand is None:
        raise ValueError("Standard deviation of demand is required and cannot be None.")
    if service_level is None:
        raise ValueError("Service level is required and cannot be None.")

    if not isinstance(review_period, (int, float)) or review_period <= 0:
        raise ValueError(f"Review period must be > 0, got {review_period}")
    if not isinstance(lead_time, (int, float)) or lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}")
    if not isinstance(avg_demand, (int, float)) or avg_demand <= 0:
        raise ValueError(f"Average demand must be > 0, got {avg_demand}")
    if not isinstance(std_demand, (int, float)) or std_demand < 0:
        raise ValueError(f"Demand std dev cannot be negative, got {std_demand}")
    if not isinstance(service_level, (int, float)) or not (0.0 < service_level < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 exclusive, got {service_level}")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "review_period": review_period,
        "lead_time": lead_time,
        "avg_demand": avg_demand,
        "std_demand": std_demand,
        "service_level": service_level,
        "current_inventory": current_inventory,
        "on_order": on_order,
        "backorders": backorders
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "P": "Review period / interval between reviews (periods)",
        "L": "Replenishment lead time (periods)",
        "P + L": "Protection period / interval requiring stockout coverage (periods)",
        "d": "Average customer demand per period (units/period)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "CSL": "Cycle Service Level / target in-stock probability",
        "z": "Standard normal safety factor = Phi^{-1}(CSL)",
        "SS": "Safety stock for protection period = z * sigma_d * sqrt(P + L) (units)",
        "M": "Target inventory / Order-up-to level = Expected Demand + SS = d * (P + L) + SS (units)",
        "IP": "Inventory Position = On_Hand + On_Order - Backorders (units)",
        "Q": "Order quantity = max(0, M - IP) (units)"
    }

    # Section 3: Standard Algebraic Formulas
    formula = {
        "protection_period": "Protection Period = P + L",
        "expected_protection_demand": "Expected Protection Demand = d * (P + L)",
        "std_protection_demand": "Std Dev Protection Demand = sigma_d * sqrt(P + L)",
        "safety_stock": "SS = z * sigma_d * sqrt(P + L)",
        "target_inventory": "Target Inventory (M) = Expected Protection Demand + SS = d * (P + L) + SS",
        "inventory_position": "Inventory Position = Current_On_Hand + On_Order - Backorders",
        "order_quantity": "Order Quantity (Q) = max(0, Target Inventory - Inventory Position)"
    }

    # 2. Intermediate calculations
    protection_period = review_period + lead_time
    z = standard_normal_inv_cdf(service_level)

    expected_protection_demand = avg_demand * protection_period
    std_protection_demand = std_demand * math.sqrt(protection_period)

    safety_stock = z * std_protection_demand
    target_inventory = expected_protection_demand + safety_stock

    inventory_position = current_inventory + on_order - backorders
    order_quantity = max(0.0, target_inventory - inventory_position)

    # Section 4: Formulas with Substituted Values
    formula_with_substituted_values = {
        "protection_period": f"Protection Period = {review_period} + {lead_time} = {protection_period:.2f} periods",
        "expected_protection_demand": (
            f"Expected Protection Demand = {avg_demand} * ({review_period} + {lead_time}) "
            f"= {avg_demand} * {protection_period:.2f} = {expected_protection_demand:.2f} units"
        ),
        "std_protection_demand": (
            f"Std Dev Protection Demand = {std_demand} * sqrt({protection_period:.2f}) "
            f"= {std_demand} * {math.sqrt(protection_period):.4f} = {std_protection_demand:.2f} units"
        ),
        "safety_stock": (
            f"SS = {z:.4f} * {std_protection_demand:.2f} = {safety_stock:.2f} units "
            f"(Rounded: {max(0, math.ceil(safety_stock)):,} units)"
        ),
        "target_inventory": (
            f"Target Inventory (M) = {expected_protection_demand:.2f} + {safety_stock:.2f} "
            f"= {target_inventory:.2f} units (Rounded: {max(0, math.ceil(target_inventory)):,} units)"
        ),
        "inventory_position": (
            f"Inventory Position = {current_inventory} + {on_order} - {backorders} = {inventory_position:.2f} units"
        ),
        "order_quantity": (
            f"Order Quantity (Q) = max(0, {target_inventory:.2f} - {inventory_position:.2f}) "
            f"= {order_quantity:.2f} units (Rounded: {max(0, math.ceil(order_quantity)):,} units)"
        )
    }

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "protection_period": round(protection_period, 2),
        "expected_demand_protection_period": round(expected_protection_demand, 2),
        "std_demand_protection_period": round(std_protection_demand, 2),
        "z_score": round(z, 4),
        "safety_stock": round(safety_stock, 2),
        "target_inventory": round(target_inventory, 2),
        "inventory_position": round(inventory_position, 2),
        "raw_order_quantity": round(order_quantity, 2)
    }

    # Section 6: Final Answer
    final_answer = {
        "order_quantity": round(order_quantity, 2),
        "optimal_order_quantity": round(order_quantity, 2),
        "recommended_order_quantity": max(0, math.ceil(order_quantity)),
        "order_quantity_exact": round(order_quantity, 2),
        "target_inventory_level": max(0, math.ceil(target_inventory)),
        "target_inventory_exact": round(target_inventory, 2),
        "safety_stock": max(0, math.ceil(safety_stock)),
        "safety_stock_exact": round(safety_stock, 2),
        "inventory_position": round(inventory_position, 2),
        "protection_period": round(protection_period, 2)
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check_name": "Protection Interval Horizon",
            "condition": "Protection period P + L > 0",
            "status": "PASS",
            "details": f"System protects across review period ({review_period}) + lead time ({lead_time}) = {protection_period:.1f} periods."
        },
        {
            "check_name": "Target Inventory Adequacy",
            "condition": "Target inventory >= Expected protection demand",
            "status": "PASS" if target_inventory >= expected_protection_demand else "FAIL",
            "details": f"Target inventory ({target_inventory:.2f}) covers expected protection demand ({expected_protection_demand:.2f}) plus buffer."
        },
        {
            "check_name": "Order Quantity Non-negativity",
            "condition": "Order quantity Q >= 0",
            "status": "PASS",
            "details": f"Calculated order quantity is {order_quantity:.2f} units."
        }
    ]

    # Section 8: Model Assumptions
    assumptions = [
        "Inventory review is conducted at fixed, discrete intervals of P periods.",
        "Stockout protection must cover the entire protection interval P + L until the subsequent order arrives.",
        "Demand across periods is independent, identically distributed, and normally distributed.",
        "Replenishment lead time L is constant and known.",
        "Inventory position is accurately accounted for as On_Hand + On_Order - Backorders.",
        "All backorders are satisfied immediately upon shipment arrival."
    ]

    return PeriodicReviewResult(
        model="Fixed-Period Review Model (Periodic Review)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        review_period=review_period,
        lead_time=lead_time,
        protection_period=round(protection_period, 2),
        protection_interval=round(protection_period, 2),
        avg_demand_per_period=avg_demand,
        std_demand_per_period=std_demand,
        service_level=service_level,
        z_score=round(z, 4),
        expected_demand_protection_interval=round(expected_protection_demand, 2),
        std_demand_protection_interval=round(std_protection_demand, 2),
        safety_stock=round(safety_stock, 2),
        safety_stock_rounded=max(0, math.ceil(safety_stock)),
        target_inventory_level=round(target_inventory, 2),
        target_inventory_level_rounded=max(0, math.ceil(target_inventory)),
        current_inventory_position=round(inventory_position, 2),
        recommended_order_quantity=round(order_quantity, 2),
        recommended_order_quantity_rounded=max(0, math.ceil(order_quantity))
    )


# Backward compatibility aliases
calculate_fixed_period_review = calculate_periodic_review
FixedPeriodResult = PeriodicReviewResult


# =============================================================================
# 5. Newsvendor Model (Single-Period Probabilistic Demand)
# =============================================================================

@dataclass
class NewsvendorResult:
    model: str
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[Dict[str, str]]
    assumptions: List[str]

    # Direct attribute access
    selling_price: float
    purchase_cost: float
    salvage_value: float
    penalty_cost: float
    underage_cost: float
    overage_cost: float
    critical_ratio: float
    critical_fractile: float  # alias
    demand_mean: float
    demand_std: float
    z_score: float
    optimal_order_quantity: float
    optimal_order_quantity_rounded: int
    recommended_order_quantity: int
    expected_lost_sales: float
    expected_sales: float
    expected_leftover_inventory: float
    expected_revenue: float
    expected_salvage_revenue: float
    expected_purchase_cost: float
    expected_penalty_cost: float
    expected_profit: float
    in_stock_probability: float

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_newsvendor(
    selling_price: float,
    purchase_cost: float,
    salvage_value: float,
    demand_mean: float,
    demand_std: float,
    penalty_cost: float = 0.0
) -> NewsvendorResult:
    """
    Calculates Underage Cost (Cu), Overage Cost (Co), Critical Ratio (CR), and Recommended Order Quantity (Q*)
    for the single-period Newsvendor model with normally distributed demand.

    Returns the complete 8-part verification breakdown:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    """
    # 1. Validate inputs (Never invent missing numerical inputs)
    if selling_price is None:
        raise ValueError("Selling price (p) is required and cannot be None.")
    if purchase_cost is None:
        raise ValueError("Purchase cost (c) is required and cannot be None.")
    if salvage_value is None:
        raise ValueError("Salvage value (s) is required and cannot be None.")
    if demand_mean is None:
        raise ValueError("Demand mean (mu) is required and cannot be None.")
    if demand_std is None:
        raise ValueError("Demand std dev (sigma) is required and cannot be None.")

    if not isinstance(selling_price, (int, float)) or selling_price <= 0:
        raise ValueError(f"Selling price must be > 0, got {selling_price}")
    if not isinstance(purchase_cost, (int, float)) or purchase_cost <= 0:
        raise ValueError(f"Purchase cost must be > 0, got {purchase_cost}")
    if not isinstance(salvage_value, (int, float)):
        raise ValueError(f"Salvage value must be a number, got {salvage_value}")
    if not isinstance(demand_mean, (int, float)) or demand_mean <= 0:
        raise ValueError(f"Demand mean must be > 0, got {demand_mean}")
    if not isinstance(demand_std, (int, float)) or demand_std <= 0:
        raise ValueError(f"Demand std dev must be > 0, got {demand_std}")
    if not isinstance(penalty_cost, (int, float)) or penalty_cost < 0:
        raise ValueError(f"Penalty cost cannot be negative, got {penalty_cost}")

    if selling_price <= purchase_cost:
        raise ValueError(f"Selling price ({selling_price}) must be strictly greater than purchase cost ({purchase_cost})")
    if purchase_cost < salvage_value:
        raise ValueError(f"Purchase cost ({purchase_cost}) should be greater than or equal to salvage value ({salvage_value})")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "selling_price": selling_price,
        "purchase_cost": purchase_cost,
        "salvage_value": salvage_value,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
        "penalty_cost": penalty_cost
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "p": "Selling price per unit ($/unit)",
        "c": "Purchase or production cost per unit ($/unit)",
        "s": "Salvage value per unsold unit ($/unit)",
        "g": "Goodwill loss or penalty cost per unsatisfied unit ($/unit)",
        "Cu": "Cost of underage / profit margin per unit = p - c + g ($/unit)",
        "Co": "Cost of overage / loss per unsold unit = c - s ($/unit)",
        "CR": "Critical Ratio / Critical Fractile = Cu / (Cu + Co)",
        "mu": "Mean customer demand (units)",
        "sigma": "Standard deviation of demand (units)",
        "z": "Standard normal safety factor = Phi^{-1}(CR)",
        "Q_star": "Optimal order quantity = mu + z * sigma (units)"
    }

    # Section 3: Standard Algebraic Formulas
    formula = {
        "underage_cost": "Cu = p - c + g",
        "overage_cost": "Co = c - s",
        "critical_ratio": "CR = Cu / (Cu + Co)",
        "z_score": "z = Phi^{-1}(CR)",
        "optimal_order_quantity": "Q* = mu + z * sigma",
        "expected_lost_sales": "Expected Lost Sales = sigma * L(z) where L(z) = phi(z) - z * (1 - Phi(z))",
        "expected_sales": "Expected Sales = mu - Expected Lost Sales",
        "expected_leftover_inventory": "Expected Leftover = Q* - Expected Sales",
        "expected_profit": "Expected Profit = (p * Expected Sales) + (s * Expected Leftover) - (c * Q*) - (g * Expected Lost Sales)"
    }

    # 2. Intermediate calculations
    underage_cost = selling_price - purchase_cost + penalty_cost
    overage_cost = purchase_cost - salvage_value
    denom = underage_cost + overage_cost
    if denom == 0:
        raise ValueError("Sum of underage cost and overage cost cannot be zero.")

    critical_ratio = underage_cost / denom
    z = standard_normal_inv_cdf(critical_ratio)
    optimal_q = demand_mean + z * demand_std
    optimal_q_rounded = max(1, round(optimal_q))

    loss_z = standard_normal_loss(z)
    expected_lost_sales = demand_std * loss_z
    expected_sales = demand_mean - expected_lost_sales
    expected_leftover = optimal_q - expected_sales

    expected_revenue = selling_price * expected_sales
    expected_salvage = salvage_value * max(0.0, expected_leftover)
    expected_total_cost = purchase_cost * optimal_q
    expected_penalty = penalty_cost * expected_lost_sales

    expected_profit = expected_revenue + expected_salvage - expected_total_cost - expected_penalty
    in_stock_prob = standard_normal_cdf(z)

    # Section 4: Formulas with Substituted Values
    formula_with_substituted_values = {
        "underage_cost": f"Cu = ${selling_price:,.2f} - ${purchase_cost:,.2f} + ${penalty_cost:,.2f} = ${underage_cost:,.2f}",
        "overage_cost": f"Co = ${purchase_cost:,.2f} - ${salvage_value:,.2f} = ${overage_cost:,.2f}",
        "critical_ratio": (
            f"CR = ${underage_cost:,.2f} / (${underage_cost:,.2f} + ${overage_cost:,.2f}) "
            f"= ${underage_cost:,.2f} / ${denom:,.2f} = {critical_ratio:.4f} ({critical_ratio*100:.2f}%)"
        ),
        "z_score": f"z = Phi^{{-1}}({critical_ratio:.4f}) = {z:.4f}",
        "optimal_order_quantity": (
            f"Q* = {demand_mean:,.0f} + ({z:.4f} * {demand_std:,.0f}) "
            f"= {demand_mean:,.0f} + {z * demand_std:,.2f} = {optimal_q:,.2f} units (Rounded: {optimal_q_rounded:,} units)"
        ),
        "expected_lost_sales": (
            f"Expected Lost Sales = {demand_std:,.0f} * L({z:.4f}) = {expected_lost_sales:,.2f} units"
        ),
        "expected_sales": (
            f"Expected Sales = {demand_mean:,.0f} - {expected_lost_sales:,.2f} = {expected_sales:,.2f} units"
        ),
        "expected_leftover_inventory": (
            f"Expected Leftover = {optimal_q:,.2f} - {expected_sales:,.2f} = {expected_leftover:,.2f} units"
        ),
        "expected_profit": (
            f"Expected Profit = (${selling_price:,.2f} * {expected_sales:,.2f}) + (${salvage_value:,.2f} * {expected_leftover:,.2f}) - (${purchase_cost:,.2f} * {optimal_q:,.2f}) "
            f"= ${expected_profit:,.2f}"
        )
    }

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "underage_cost": round(underage_cost, 2),
        "overage_cost": round(overage_cost, 2),
        "critical_ratio": round(critical_ratio, 4),
        "z_score": round(z, 4),
        "expected_lost_sales": round(expected_lost_sales, 2),
        "expected_sales": round(expected_sales, 2),
        "expected_leftover_inventory": round(expected_leftover, 2),
        "expected_revenue": round(expected_revenue, 2),
        "expected_salvage_revenue": round(expected_salvage, 2),
        "expected_purchase_cost": round(expected_total_cost, 2),
        "expected_penalty_cost": round(expected_penalty, 2),
        "expected_profit": round(expected_profit, 2),
        "in_stock_probability": round(in_stock_prob, 4)
    }

    # Section 6: Final Answer
    final_answer = {
        "underage_cost": round(underage_cost, 2),
        "overage_cost": round(overage_cost, 2),
        "critical_ratio": round(critical_ratio, 4),
        "optimal_order_quantity": round(optimal_q, 2),
        "recommended_order_quantity": optimal_q_rounded,
        "optimal_order_quantity_exact": round(optimal_q, 2),
        "expected_maximum_profit": round(expected_profit, 2),
        "in_stock_probability": round(in_stock_prob, 4),
        "expected_sales": round(expected_sales, 2),
        "expected_lost_sales": round(expected_lost_sales, 2),
        "expected_leftover_inventory": round(expected_leftover, 2)
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check_name": "Critical Ratio Unit Bounds",
            "condition": "0 < Critical Ratio < 1.0",
            "status": "PASS" if 0.0 < critical_ratio < 1.0 else "FAIL",
            "details": f"Critical ratio ({critical_ratio:.4f}) is strictly bounded between 0 and 1."
        },
        {
            "check_name": "Economic Profitability Condition",
            "condition": "Selling Price (p) > Purchase Cost (c) >= Salvage Value (s)",
            "status": "PASS" if selling_price > purchase_cost >= salvage_value else "WARNING",
            "details": f"Price (${selling_price:,.2f}) > Cost (${purchase_cost:,.2f}) >= Salvage (${salvage_value:,.2f})."
        },
        {
            "check_name": "Optimal Quantity Positivity",
            "condition": "Optimal Q* > 0",
            "status": "PASS" if optimal_q > 0 else "FAIL",
            "details": f"Recommended order quantity ({optimal_q_rounded:,} units) is positive."
        }
    ]

    # Section 8: Model Assumptions
    assumptions = [
        "Single, non-replenishable planning season (perishable/fashion/seasonal products).",
        "Customer demand is uncertain and follows a continuous normal distribution.",
        "Unsold inventory at the end of the season is liquidated at constant salvage value s.",
        "Unsatisfied customer demand represents lost sales with optional goodwill penalty cost g.",
        "The decision objective is to maximize expected total net profit."
    ]

    return NewsvendorResult(
        model="Newsvendor Model (Single-Period)",
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        selling_price=selling_price,
        purchase_cost=purchase_cost,
        salvage_value=salvage_value,
        penalty_cost=penalty_cost,
        underage_cost=round(underage_cost, 2),
        overage_cost=round(overage_cost, 2),
        critical_ratio=round(critical_ratio, 4),
        critical_fractile=round(critical_ratio, 4),
        demand_mean=demand_mean,
        demand_std=demand_std,
        z_score=round(z, 4),
        optimal_order_quantity=round(optimal_q, 2),
        optimal_order_quantity_rounded=optimal_q_rounded,
        recommended_order_quantity=optimal_q_rounded,
        expected_lost_sales=round(expected_lost_sales, 2),
        expected_sales=round(expected_sales, 2),
        expected_leftover_inventory=round(expected_leftover, 2),
        expected_revenue=round(expected_revenue, 2),
        expected_salvage_revenue=round(expected_salvage, 2),
        expected_purchase_cost=round(expected_total_cost, 2),
        expected_penalty_cost=round(expected_penalty, 2),
        expected_profit=round(expected_profit, 2),
        in_stock_probability=round(in_stock_prob, 4)
    )


def generate_newsvendor_curve(
    selling_price: float,
    purchase_cost: float,
    salvage_value: float,
    demand_mean: float,
    demand_std: float,
    penalty_cost: float = 0.0,
    points: int = 40
) -> List[Dict[str, float]]:
    """Generates expected profit vs order quantity data for visualization."""
    min_q = max(1.0, demand_mean - 3.0 * demand_std)
    max_q = demand_mean + 3.0 * demand_std
    step = (max_q - min_q) / (points - 1)

    curve = []
    for i in range(points):
        q = min_q + i * step
        z = (q - demand_mean) / demand_std
        loss_z = standard_normal_loss(z)
        exp_lost = demand_std * loss_z
        exp_sales = demand_mean - exp_lost
        exp_leftover = q - exp_sales
        exp_profit = (
            (selling_price * exp_sales)
            + (salvage_value * exp_leftover)
            - (purchase_cost * q)
            - (penalty_cost * exp_lost)
        )
        curve.append({
            "order_quantity": round(q, 1),
            "expected_profit": round(exp_profit, 2),
            "expected_sales": round(exp_sales, 1),
            "expected_leftover": round(exp_leftover, 1)
        })
    return curve


# =============================================================================
# CSV Data Analysis Utilities
# =============================================================================

REQUIRED_CSV_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]


def validate_inventory_csv_columns(df: Any) -> List[str]:
    """
    Validates that required columns (Product, Category, Units, Unit Cost) exist.
    Returns a list of missing column names. Empty list means all columns are present.
    """
    return [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]


def analyze_inventory_dataframe(df_input: Any) -> Dict[str, Any]:
    """
    Performs deterministic pandas analytics on the uploaded inventory DataFrame:
    1. Validates required columns (Product, Category, Units, Unit Cost).
    2. Calculates total Units.
    3. Calculates average Unit Cost.
    4. Identifies the Product with the highest Units.
    5. Groups data by Category and calculates total Units for each Category.
    6. Adds new calculated column: Inventory Value = Units * Unit Cost.
    7. Calculates total Inventory Value across all products.

    Returns a structured dictionary containing all calculations and the augmented DataFrame.
    """
    import pandas as pd

    missing = validate_inventory_csv_columns(df_input)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df_input.copy()

    # Clean numeric fields if strings/commas/currency symbols exist
    if not pd.api.types.is_numeric_dtype(df["Units"]):
        df["Units"] = df["Units"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0)

    if not pd.api.types.is_numeric_dtype(df["Unit Cost"]):
        df["Unit Cost"] = df["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce").fillna(0.0)

    # 1. Total Units
    total_units = float(df["Units"].sum())

    # 2. Average Unit Cost
    avg_unit_cost = float(df["Unit Cost"].mean()) if len(df) > 0 else 0.0

    # 3. Product with highest Units
    if len(df) > 0:
        idx_max = df["Units"].idxmax()
        highest_units_product = str(df.loc[idx_max, "Product"])
        highest_units_count = float(df.loc[idx_max, "Units"])
    else:
        highest_units_product = "None"
        highest_units_count = 0.0

    # 4. Group data by Category and calculate total Units for each Category
    category_group = (
        df.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
        .sort_values(by="Total Units", ascending=False)
        .reset_index(drop=True)
    )

    # 5. Create new calculated column: Inventory Value = Units * Unit Cost
    df["Inventory Value"] = df["Units"] * df["Unit Cost"]

    # 6. Calculate total Inventory Value across all products
    total_inventory_value = float(df["Inventory Value"].sum())

    return {
        "df": df,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "product_highest_units": highest_units_product,
        "highest_units_count": highest_units_count,
        "total_units_by_category": category_group,
        "total_inventory_value": total_inventory_value
    }

